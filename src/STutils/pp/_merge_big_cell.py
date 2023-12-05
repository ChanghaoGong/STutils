import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import BisectingKMeans


def merge_big_cell(
    adata: AnnData,
    resolution: str,
    prefix: str,
    merge_tags: list[str],
    n: int = 30,
) -> AnnData:
    """Merge STomics cellbin data to metacell by axis and celltype.

    Args:
        adata (AnnData): adata object
        resolution (str): celltype tag in adata.obs
        prefix (str): prefix of output file
        merge_tags (list[str]): tags to merge
        n (int, optional): cell numbers to merge, defaults to 30

    Returns
    -------
        AnnData: merged adata object with resulution and merge_tags
    """
    adata_list = []
    for category in adata.obs[resolution].unique():
        sub_adata = adata[adata.obs[resolution] == category].copy()
        adata_list.append(sub_adata)
    merged_adatas = []
    merged_cluster_labels = pd.DataFrame(columns=["cell_id", "merged_cluster"])
    for i, st_adata in enumerate(adata_list):
        t = st_adata.shape[0] // n
        X = st_adata.obs[["x", "y"]].values
        spectral_clustering = BisectingKMeans(n_clusters=t, bisecting_strategy="largest_cluster", random_state=0)
        cluster_labels = spectral_clustering.fit_predict(X)
        # cluster_sizes = np.bincount(cluster_labels)
        # sorted_clusters = np.argsort(cluster_sizes)[::-1]
        # max_cluster_index = sorted_clusters[0]
        # while max_cluster_index
        cluster = st_adata.obs[resolution].iloc[0]
        adata_idx = adata.obs[resolution].unique().to_list().index(cluster)
        merged_cluster_labels_i = pd.DataFrame(
            list(zip(st_adata.obs_names, [f"{i-1}_{adata_idx}" for i in cluster_labels])),
            columns=["cell_id", "merged_cluster"],
        )
        merged_cluster_labels = pd.concat([merged_cluster_labels, merged_cluster_labels_i], axis=0)
        # 计算每个簇的中心坐标和gene count
        cluster_centers = []
        cluster_gene_counts = []
        cluster_cell_counts = []
        for cluster_id in np.unique(cluster_labels):
            cluster_spots = st_adata.obs.index[cluster_labels == cluster_id]
            cluster_center = np.mean(st_adata.obs.loc[cluster_spots, ["x", "y"]], axis=0)
            cluster_gene_count = np.sum(st_adata[cluster_spots].X, axis=0)
            cluster_centers.append(cluster_center)
            cluster_gene_counts.append(cluster_gene_count)
            cluster_cell_counts.append(cluster_spots.shape[0])
        # 构建合并后的adata对象
        adata_cluster = sc.AnnData(
            X=np.array(cluster_gene_counts)[:, 0, :],
            obsm={"spatial": np.array(cluster_centers)},
            obs=pd.DataFrame(
                cluster_cell_counts, index=np.arange(len(cluster_centers)), columns=["cluster_cell_counts"]
            ),
            var=adata.var,
        )
        adata_cluster.obs[resolution] = st_adata.obs[resolution].iloc[0]
        for tag in merge_tags:
            adata_cluster.obs[tag] = st_adata.obs[tag].iloc[0]
        merged_adatas.append(adata_cluster)
    merged_cluster_labels.to_csv(f"merged_cluster_labels_{prefix}.txt", sep="\t", index=None)
    # 合并所有合并后的adata对象
    merged_adata = sc.AnnData.concatenate(*merged_adatas, join="outer")
    return merged_adata
