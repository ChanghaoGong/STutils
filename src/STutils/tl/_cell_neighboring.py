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
    """Calculate nhood enrichment score.

    Args:
        adata (AnnData): spatial anndata
        coord_type (str, optional): Type of coordinate system. Defaults to 'generic'.
        library_key (str, optional): batch info. Defaults to 'batch'.
        radius (int, optional): Compute the graph based on neighborhood radius. Defaults to 30.
        cluster_key (str, optional): region or cell cluster key. Defaults to "region".

    Returns
    -------
        pd.DataFrame: a dataframe of neighborhood enrichment
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
