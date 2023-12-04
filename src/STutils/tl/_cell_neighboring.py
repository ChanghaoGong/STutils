import pandas as pd
import squidpy as sq
from anndata import AnnData


def nhood_enrichment(
    adata: AnnData,
    coord_type: str = "generic",
    library_key: str = "batch",
    radius: float = 30,
    cluster_key: str = "region",
) -> pd.DataFrame:
    """_summary_

    :param adata: spatial anndata
    :type adata: AnnData
    :param coord_type: Type of coordinate system., defaults to "generic"
    :type coord_type: str, optional
    :param library_key:batch info, defaults to "batch"
    :type library_key: str, optional
    :param radius: Compute the graph based on neighborhood radius, defaults to 30
    :type radius: float, optional
    :param cluster_key: region or cell cluster key, defaults to "region"
    :type cluster_key: str, optional
    :return:a dataframe of neighborhood enrichment
    :rtype: pd.DataFrame
    """
    sq.gr.spatial_neighbors(adata, coord_type=coord_type, library_key=library_key, radius=radius)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
    region_number = adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].cat.categories]
    nhood_counts = pd.DataFrame(
        adata.uns[f"{cluster_key}_nhood_enrichment"]["count"],
        index=adata.obs[cluster_key].cat.categories,
        columns=adata.obs[cluster_key].cat.categories,
    )
    nhood_percents = nhood_counts / region_number
    return nhood_percents
