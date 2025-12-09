"""Plotting for nhood heatmap."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from joblib import Parallel, delayed
from matplotlib.axes import Axes

from STutils.tl import nhood_enrichment, summarise_nhood


def nhood_heatmap(
    adata: AnnData,
    coord_type: str = "generic",
    library_key: str = "batch",
    radius: float = 30,
    cluster_key: str = "region",
    ax: Axes | None = None,
    figsize: tuple = (6, 5),
    cmap: str = "YlGn",
    save: bool = True,
) -> Axes:
    """Plot neighborhood heatmap.

    Args:
        adata: anndata object
        coord_type: Type of coordinate system, defaults to "generic"
        library_key: batch info, defaults to "batch"
        radius: Compute the graph based on neighborhood radius, defaults to 30
        cluster_key: region or cell cluster key, defaults to "region"
        ax: mpl axes, defaults to None
        figsize: fig size, defaults to (6, 5)
        cmap: colormap, defaults to "YlGn"
        save: save or not, defaults to True

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


def plot_summary(
    list_nhood: dict[str, pd.DataFrame],
    cell_types: list[str],
    cell_type_column: str = "cell_type_tidy",
    excluded_types: list[str] | None = None,
    n_jobs: int = 6,
) -> pd.DataFrame:
    """
    汇总所有样本的邻域分析结果并准备绘图数据

    参数:
        list_nhood: 样本名到邻域分析结果的字典
        cell_types: 包含所有细胞类型的列表
        cell_type_column: 细胞类型列名
        excluded_types: 要排除的细胞类型列表
        n_jobs: 并行工作数

    返回:
        汇总后的DataFrame
    """
    if excluded_types is None:
        excluded_types = ["excluded"]
    # 并行处理所有样本
    results = Parallel(n_jobs=n_jobs)(
        delayed(summarise_nhood)(sample_df) for sample, sample_df in list_nhood.items()
    )

    # 合并所有结果
    nhood_summary = pd.concat(results)

    # 汇总统计
    summary_df = (
        nhood_summary.groupby(
            ["from_cell_type_figure", "to_cell_type_figure", "interaction_type"]
        )["z_score"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # 对每个细胞类型对保留最常见的相互作用类型
    summary_df = summary_df.sort_values("count", ascending=False).drop_duplicates(
        ["from_cell_type_figure", "to_cell_type_figure"]
    )

    # 创建完整的细胞类型组合网格
    from_types = [ct for ct in cell_types if ct not in excluded_types]
    to_types = [ct for ct in cell_types if ct not in excluded_types]

    full_grid = pd.MultiIndex.from_product(
        [from_types, to_types], names=["from_cell_type_figure", "to_cell_type_figure"]
    ).to_frame(index=False)

    # 合并完整网格与结果
    nhood_summary = full_grid.merge(
        summary_df, on=["from_cell_type_figure", "to_cell_type_figure"], how="left"
    )

    # 计算样本百分比(假设总样本数为12)
    nhood_summary["percent_samples"] = nhood_summary["count"] / 12

    return nhood_summary


def plot_interaction_heatmap(
    summary_df: pd.DataFrame,
    title: str = "Cell-Cell Interaction Summary",
    figsize: tuple = (12, 10),
    cmap: str = "coolwarm",
    vmin: float = -2,
    vmax: float = 2,
    annot: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    绘制细胞-细胞相互作用热图

    参数:
        summary_df: 来自plot_summary的结果DataFrame
        title: 图标题
        figsize: 图大小
        cmap: 颜色映射
        vmin: 颜色条最小值
        vmax: 颜色条最大值
        annot: 是否显示数值
        fmt: 数值格式
    """
    # 创建数据透视表
    pivot_df = summary_df.pivot(
        index="from_cell_type_figure", columns="to_cell_type_figure", values="mean"
    )

    # 绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        center=0,
        linewidths=0.5,
        linecolor="lightgray",
    )

    plt.title(title)
    plt.xlabel("To Cell Type")
    plt.ylabel("From Cell Type")
    plt.tight_layout()
    plt.show()
