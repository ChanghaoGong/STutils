"""Plotting for nhood heatmap."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes

from STutils.tl import nhood_enrichment


def nhood_heatmap(
    adata: AnnData,
    coord_type: str = "generic",
    library_key: str = "batch",
    radius: float = 30,
    cluster_key: str = "region",
    ax: Optional[Axes] = None,
    figsize: tuple = (6, 5),
    cmap: str = "YlGn",
    save: bool = True,
) -> Axes:
    """Plot neighborhood heatmap.

    Args:
        adata (AnnData): anndata object
        coord_type (str, optional): Type of coordinate system, defaults
            to "generic"
        library_key (str, optional): batch info, defaults to "batch"
        radius (float, optional): Compute the graph based on
            neighborhood radius, defaults to 30
        cluster_key (str, optional): region or cell cluster key,
            defaults to "region"
        ax (Optional[Axes], optional): mpl axes, defaults to None
        figsize (tuple, optional): fig size, defaults to (6, 5)
        cmap (str, optional): colormap, defaults to "YlGn"
        save (bool, optional): save or not, defaults to True

    Returns
    -------
        Axes: mpl axes
    """
    nhood_percents = nhood_enrichment(
        adata,
        coord_type=coord_type,
        library_key=library_key,
        radius=radius,
        cluster_key=cluster_key,
    )
    # Remove the numbers on the diagonal
    nhood_percents = nhood_percents.mask(np.eye(len(nhood_percents), dtype=bool))
    # the numbers on the diagonal equal to max value in the matrix
    nhood_percents = nhood_percents.mask(
        np.eye(len(nhood_percents), dtype=bool), nhood_percents.max().max()
    )
    # zscore nhood_percents
    # nhood_percents = (nhood_percents - nhood_percents.mean()) / nhood_percents.std()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(nhood_percents, cmap=cmap, linewidth=0.3, linecolor="#929292", ax=ax)
    # set the NA value as grey color
    ax.set_facecolor("#dcdcdc")
    if save:
        outfig = f"{cluster_key}_nhood_heatmap.pdf"
        plt.savefig(outfig, dpi=300, format="pdf", bbox_inches="tight")
    return ax
