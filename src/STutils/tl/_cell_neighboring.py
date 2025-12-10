from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from pandas import CategoricalDtype
from pandas.api.types import infer_dtype
from scipy.sparse import block_diag, csr_matrix
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale


def _assert_categorical_obs(adata: AnnData, key: str) -> None:
    """Assert that a key in adata.obs is categorical."""
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")
    if not isinstance(adata.obs[key].dtype, CategoricalDtype):
        raise TypeError(
            f"Expected `adata.obs[{key!r}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`."
        )


def _assert_spatial_basis(adata: AnnData, key: str) -> None:
    """Assert that spatial coordinates exist in adata.obsm."""
    if key not in adata.obsm:
        raise KeyError(f"Spatial basis `{key}` not found in `adata.obsm`.")


def _build_connectivity(
    coords: np.ndarray,
    n_neighs: int = 6,
    radius: float | None = None,
    delaunay: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    """Build connectivity graph from spatial coordinates.

    Simplified version of squidpy.gr._build._build_connectivity.
    """
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix(
            (np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N)
        )

        if return_distance:
            dists = np.array(
                list(
                    chain(
                        *(
                            euclidean_distances(
                                coords[indices[indptr[i] : indptr[i + 1]], :],
                                coords[np.newaxis, i, :],
                            )
                            for i in range(N)
                            if len(indices[indptr[i] : indptr[i + 1]])
                        )
                    )
                )
            ).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
    else:
        r = 1 if radius is None else radius
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)),
            shape=(N, N),
        )
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        Adj.eliminate_zeros()
        Dst.eliminate_zeros()
        return Adj, Dst

    Adj.eliminate_zeros()
    return Adj


def _spatial_neighbors_simplified(
    adata: AnnData,
    spatial_key: str = "spatial",
    coord_type: str = "generic",
    library_key: str | None = None,
    radius: float | None = None,
    key_added: str = "spatial",
) -> None:
    """Simplified version of squidpy.gr.spatial_neighbors.

    Only supports coord_type="generic" with radius parameter.
    """
    _assert_spatial_basis(adata, spatial_key)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libs = adata.obs[library_key].cat.categories
        mats: list[tuple[csr_matrix, csr_matrix]] = []
        ixs: list[int] = []
        for lib in libs:
            ixs.extend(np.where(adata.obs[library_key] == lib)[0])
            adata_subset = adata[adata.obs[library_key] == lib]
            Adj, Dst = _build_connectivity(
                adata_subset.obsm[spatial_key],
                n_neighs=6,
                radius=radius,
                delaunay=False,
                return_distance=True,
                set_diag=False,
            )
            mats.append((Adj, Dst))
        ixs = np.argsort(ixs).tolist()
        Adj = block_diag([m[0] for m in mats], format="csr")[ixs, :][:, ixs]
        Dst = block_diag([m[1] for m in mats], format="csr")[ixs, :][:, ixs]
    else:
        Adj, Dst = _build_connectivity(
            adata.obsm[spatial_key],
            n_neighs=6,
            radius=radius,
            delaunay=False,
            return_distance=True,
            set_diag=False,
        )

    conns_key = f"{key_added}_connectivities"
    dists_key = f"{key_added}_distances"
    adata.obsp[conns_key] = Adj
    adata.obsp[dists_key] = Dst


def _count_nhood_enrichment(
    adj: csr_matrix, clustering: np.ndarray, n_cls: int
) -> np.ndarray:
    """Count neighborhood enrichment without numba.

    Simplified pure Python version.
    """
    res = np.zeros((adj.shape[0], n_cls), dtype=np.uint32)
    for i in range(adj.shape[0]):
        xs, xe = adj.indptr[i], adj.indptr[i + 1]
        cols = adj.indices[xs:xe]
        for c in cols:
            res[i, clustering[c]] += 1

    # Aggregate by cluster
    count_matrix = np.zeros((n_cls, n_cls), dtype=np.uint32)
    for row in range(res.shape[0]):
        cl = clustering[row]
        count_matrix[cl, :] += res[row, :]

    return count_matrix


def _nhood_enrichment_simplified(
    adata: AnnData,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    seed: int | None = None,
) -> None:
    """Simplified version of squidpy.gr.nhood_enrichment.

    Uses pure Python implementation instead of numba.
    """
    if connectivity_key is None:
        connectivity_key = "spatial_connectivities"
    else:
        connectivity_key = f"{connectivity_key}_connectivities"

    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Spatial connectivity key `{connectivity_key}` not found in `adata.obsp`. "
            f"Please run `spatial_neighbors` first."
        )

    _assert_categorical_obs(adata, cluster_key)
    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=np.uint32)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libraries: pd.Series | None = adata.obs[library_key]
    else:
        libraries = None

    n_cls = len(clust_map)
    count = _count_nhood_enrichment(adj, int_clust, n_cls)

    # Permutation test
    rs = np.random.RandomState(seed)
    perms = np.empty((n_perms, n_cls, n_cls), dtype=np.float64)
    int_clust_copy = int_clust.copy()

    for i in range(n_perms):
        if libraries is not None:
            # Shuffle within each library
            int_clust_shuff = int_clust_copy.copy()
            for c in libraries.cat.categories:
                idx = np.where(libraries == c)[0]
                arr_group = int_clust_shuff[idx].copy()
                rs.shuffle(arr_group)
                int_clust_shuff[idx] = arr_group
            int_clust_perm = int_clust_shuff
        else:
            int_clust_perm = int_clust_copy.copy()
            rs.shuffle(int_clust_perm)

        perms[i, ...] = _count_nhood_enrichment(adj, int_clust_perm, n_cls)

    zscore = (count - perms.mean(axis=0)) / (perms.std(axis=0) + 1e-10)

    adata.uns[f"{cluster_key}_nhood_enrichment"] = {"zscore": zscore, "count": count}


def nhood_enrichment(
    adata: AnnData,
    coord_type: str = "generic",
    library_key: str = "batch",
    radius: float = 30,
    cluster_key: str = "region",
) -> pd.DataFrame:
    """Calculate neighborhood enrichment.

    Args:
        adata: spatial anndata
        coord_type: Type of coordinate system, defaults to "generic"
        library_key: batch info, defaults to "batch"
        radius: Compute the graph based on neighborhood radius, defaults to 30
        cluster_key: region or cell cluster key, defaults to "region"

    Returns
    -------
        pd.DataFrame: a dataframe of neighborhood enrichment
    """
    _spatial_neighbors_simplified(
        adata, coord_type=coord_type, library_key=library_key, radius=radius
    )
    _nhood_enrichment_simplified(
        adata, cluster_key=cluster_key, library_key=library_key
    )
    region_number = adata.obs[cluster_key].value_counts()[
        adata.obs[cluster_key].cat.categories
    ]
    nhood_counts = pd.DataFrame(
        adata.uns[f"{cluster_key}_nhood_enrichment"]["count"],
        index=adata.obs[cluster_key].cat.categories,
        columns=adata.obs[cluster_key].cat.categories,
    )
    nhood_percents = nhood_counts / region_number
    return nhood_percents


def test_nhood(
    adata: AnnData,
    cell_type_column: str = "cell_type",
    spatial_key: str = "spatial",
    radius: float = 27.5,
    fraction_coherence: float = 0.8,
    iterations: int = 200,
    workers: int = 10,
    excluded_types: list[str] | None = None,
    sample_key: str | None = None,
    sample_id: str | None = None,
) -> pd.DataFrame:
    """
    Analyze cell type neighborhood interactions with spatial shuffling test.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data with spatial coordinates
    cell_type_column : str
        Column name in adata.obs containing cell type information
    spatial_key : str
        Key in adata.obsm containing spatial coordinates (default: 'spatial')
    radius : float
        Radius to define neighborhood (default: 27.5)
    fraction_coherence : float
        Threshold for filtering cells with dominant neighbor type (default: 0.8)
    iterations : int
        Number of shuffling iterations (default: 10)
    workers : int
        Number of parallel workers (default: 10)
    excluded_types : List[str]
        List of cell types to exclude from analysis (default: [])
    sample_key : Optional[str]
        Column name in adata.obs containing sample IDs (optional)
    sample_id : Optional[str]
        Specific sample ID to analyze (required if sample_key is provided)

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized observed and shuffled neighborhood interactions
    """
    if excluded_types is None:
        excluded_types = []
    # Validate inputs
    if cell_type_column not in adata.obs.columns:
        raise ValueError(
            f"Cell type column '{cell_type_column}' not found in adata.obs"
        )

    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates key '{spatial_key}' not found in adata.obsm"
        )

    if sample_key is not None:
        if sample_key not in adata.obs.columns:
            raise ValueError(f"Sample key '{sample_key}' not found in adata.obs")
        if sample_id is None:
            raise ValueError("sample_id must be provided when sample_key is specified")
        if sample_id not in adata.obs[sample_key].unique():
            raise ValueError(
                f"Sample ID '{sample_id}' not found in column '{sample_key}'"
            )

    # Subset data if sample information is provided
    if sample_key is not None:
        adata = adata[adata.obs[sample_key] == sample_id].copy()

    # Filter out excluded cell types
    adata = adata[~adata.obs[cell_type_column].isin(excluded_types)].copy()

    # Observed part
    def calculate_observed(adata: AnnData) -> pd.DataFrame:
        # Compute neighborhood graph
        _spatial_neighbors_simplified(
            adata, coord_type="generic", radius=radius, key_added="expansion_graph"
        )

        # Get adjacency matrix
        adj_matrix = adata.obsp["expansion_graph_connectivities"]

        # Create neighborhood matrix
        cell_types = adata.obs[cell_type_column].unique().tolist()
        nhood_mat = pd.DataFrame(0, index=adata.obs.index, columns=cell_types)

        for i, cell in enumerate(adata.obs.index):
            neighbors = adj_matrix[i].nonzero()[1]
            neighbor_types = adata.obs.iloc[neighbors][cell_type_column]
            counts = neighbor_types.value_counts()
            nhood_mat.loc[cell, counts.index] = counts

        nhood_mat[f"from_{cell_type_column}"] = adata.obs[cell_type_column].values

        # Filter based on fraction coherence
        row_sums = nhood_mat[cell_types].sum(axis=1)
        fraction_matrix = nhood_mat[cell_types].div(row_sums, axis=0)
        drop_rows = (fraction_matrix > fraction_coherence).any(axis=1)
        nhood_norm_coherent = nhood_mat[~drop_rows].dropna()
        # Summarize interactions
        nhood_sum = nhood_norm_coherent.groupby(f"from_{cell_type_column}")[
            cell_types
        ].sum()

        # Normalize by row sums minus self-pairs
        nhood_row_sums = nhood_sum.sum(axis=1)
        self_pair_counts = pd.Series(
            {ct: nhood_sum.loc[ct, ct] for ct in nhood_sum.index}
        )

        nhood_norm = nhood_sum.div(nhood_row_sums - self_pair_counts, axis=0)
        nhood_norm = nhood_norm.reset_index().melt(
            id_vars=f"from_{cell_type_column}",
            var_name=f"to_{cell_type_column}",
            value_name="obs_count",
        )

        return nhood_norm

    # Shuffled part
    def calculate_shuffled(adata: AnnData) -> list[pd.DataFrame]:
        # 修改这里：将adata作为闭包变量捕获，而不是作为参数传递
        def single_shuffle(iter_num: int) -> pd.DataFrame:
            # 使用外部函数的adata参数
            adata_shuff = adata.copy()
            adata_shuff.obs[cell_type_column] = np.random.permutation(
                adata_shuff.obs[cell_type_column].values
            )

            # 计算邻域图
            _spatial_neighbors_simplified(
                adata_shuff,
                coord_type="generic",
                radius=radius,
                key_added="expansion_graph",
            )

            # 获取邻接矩阵
            adj_matrix = adata_shuff.obsp["expansion_graph_connectivities"]
            cell_types = adata_shuff.obs[cell_type_column].unique().tolist()

            # 创建邻域矩阵
            nhood_mat = pd.DataFrame(0, index=adata_shuff.obs.index, columns=cell_types)

            for i, cell in enumerate(adata_shuff.obs.index):
                neighbors = adj_matrix[i].nonzero()[1]
                neighbor_types = adata_shuff.obs.iloc[neighbors][cell_type_column]
                counts = neighbor_types.value_counts()
                for ct in counts.index:
                    nhood_mat.loc[cell, ct] = counts[ct]

            nhood_mat[f"from_{cell_type_column}"] = adata_shuff.obs[
                cell_type_column
            ].values

            # 过滤
            row_sums = nhood_mat[cell_types].sum(axis=1)
            fraction_matrix = nhood_mat[cell_types].div(row_sums, axis=0)
            drop_rows = (fraction_matrix > fraction_coherence).any(axis=1)
            nhood_norm_coherent = nhood_mat[~drop_rows].dropna()
            # 汇总
            nhood_sum = nhood_norm_coherent.groupby(f"from_{cell_type_column}")[
                cell_types
            ].sum()

            # 标准化
            nhood_row_sums = nhood_sum.sum(axis=1)
            self_pair_counts = pd.Series(
                {
                    ct: nhood_sum.loc[ct, ct] if ct in nhood_sum.index else 0
                    for ct in cell_types
                }
            )

            nhood_norm = nhood_sum.div(nhood_row_sums - self_pair_counts, axis=0)
            nhood_norm = nhood_norm.reset_index().melt(
                id_vars=f"from_{cell_type_column}",
                var_name=f"to_{cell_type_column}",
                value_name=f"shuff_{iter_num}",
            )

            return nhood_norm

        # Run in parallel
        results = Parallel(n_jobs=workers)(
            delayed(single_shuffle)(i) for i in range(iterations)
        )
        return results

    # Calculate observed and shuffled results
    nhood_norm_obs = calculate_observed(adata)
    list_nhood_norm_shuff = calculate_shuffled(adata)

    # Merge all results
    nhood_norm_shuff = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=[f"from_{cell_type_column}", f"to_{cell_type_column}"],
            how="left",
        ),
        [nhood_norm_obs] + list_nhood_norm_shuff,
    )

    # Scale the numeric columns
    numeric_cols = nhood_norm_shuff.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaled_values = scale(nhood_norm_shuff[numeric_cols].values.T).T
        nhood_norm_shuff[numeric_cols] = scaled_values

    return nhood_norm_shuff


def summarise_nhood(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总单个样本的邻域分析结果

    参数:
        sample_df: 包含邻域分析结果的DataFrame

    返回:
        汇总后的DataFrame
    """
    # 复制数据避免修改原数据
    tmp = sample_df.copy()

    # 获取观察值列名(假设第三列是观察值)
    obs_col = tmp.columns[2]

    # 计算置换测试的统计量
    shuff_cols = [col for col in tmp.columns if "shuff_" in col]

    tmp["z_score"] = tmp[obs_col]
    tmp["perm_min"] = tmp[shuff_cols].min(axis=1)
    tmp["perm_max"] = tmp[shuff_cols].max(axis=1)
    tmp["perm_mean"] = tmp[shuff_cols].mean(axis=1)
    tmp["perm_median"] = tmp[shuff_cols].median(axis=1)
    tmp["perm_sd"] = tmp[shuff_cols].std(axis=1)

    # 计算大于和小于观察值的次数
    tmp["count_larger"] = (tmp[shuff_cols] > tmp[obs_col]).sum(axis=1)
    tmp["count_smaller"] = (tmp[shuff_cols] < tmp[obs_col]).sum(axis=1)

    # 判断相互作用类型和显著性
    tmp["interaction_type"] = np.where(tmp[obs_col] > 0, "attraction", "avoidance")
    tmp["p_value"] = np.where(
        tmp[obs_col] > 0,
        1 / (tmp["count_smaller"] + 1),
        1 / (tmp["count_larger"] + 1),  # 加1避免除以0
    )
    tmp["significant"] = tmp["p_value"] < 0.05
    tmp["pair"] = tmp["from_cell_type_figure"] + "_" + tmp["to_cell_type_figure"]

    # 选择需要的列
    result_cols = [
        "from_cell_type_figure",
        "to_cell_type_figure",
        "z_score",
        "interaction_type",
        "p_value",
        "significant",
        "perm_mean",
        "perm_median",
        "perm_min",
        "perm_max",
        "perm_sd",
        "count_larger",
        "count_smaller",
        "pair",
    ]

    return tmp[result_cols]
