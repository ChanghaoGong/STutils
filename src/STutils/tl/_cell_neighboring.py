from functools import reduce

import numpy as np
import pandas as pd
import squidpy as sq
from anndata import AnnData
from joblib import Parallel, delayed
from sklearn.preprocessing import scale


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
    sq.gr.spatial_neighbors(
        adata, coord_type=coord_type, library_key=library_key, radius=radius
    )
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
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
        # Compute neighborhood graph with squidpy
        sq.gr.spatial_neighbors(
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
            sq.gr.spatial_neighbors(
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
