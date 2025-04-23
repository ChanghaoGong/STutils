from typing import Literal, Optional, Union

import cv2
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar

from ._utils import crop_to_square_with_padding, getDefaultColors, hex_to_rgb, int_to_rgb_idx


def cell_bin_plot(mask: str, res: pd.DataFrame, tag: str, colors: Optional[Union[list, dict[str, str]]]) -> np.ndarray:
    """Cell bin plot for stereo-seq data

    Args:
        mask (str): mask tif file path
        res (pd.DataFrame): obs dataframe of cell bin, three columns: x, y, tag
        tag (str): adata.obs.columns to plot
        colors (Sequence): color list corresponding to tag

    Returns
    -------
        np.ndarray: colored cellbin image array
    """
    # 读取二进制掩码图像
    image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    # 设置4邻域连接
    connectivity = 4
    # 使用connectedComponentsWithStats进行联通区域分析
    output = cv2.connectedComponentsWithStats(image, connectivity=connectivity)

    # 获取聚类标签和颜色
    clusters = res[tag].unique()
    # colors = getDefaultColors(cluster_number, type=colors)

    # 为每个聚类标签添加颜色列
    if isinstance(colors, list):
        res["color"] = res[tag].apply(lambda x: colors[clusters.tolist().index(x)])
    elif isinstance(colors, dict):
        res["color"] = res[tag].apply(lambda x: colors[x])

    # 将'x'和'y'从float类型转换为int类型
    res["x"] = res["x"].astype(int)
    res["y"] = res["y"].astype(int)

    # 将每个像素的联通区域标签添加到res中
    res["connectedComponentsLabel"] = output[1][res["y"].values, res["x"].values]

    # 创建一个颜色映射，将每个联通区域标签映射到对应的颜色
    colorMap = {
        connectedComponentsLabel: hex_to_rgb(color)
        for connectedComponentsLabel, color in zip(res["connectedComponentsLabel"].values, res["color"].values)
    }
    # 将背景（标签为0）的颜色设为黑色
    colorMap[0] = (0, 0, 0)

    # 将联通区域标签映射为颜色的索引
    res_idx = int_to_rgb_idx(output[1], colorMap)

    # 将索引映射为RGB图像
    image = np.uint8(res_idx)

    return image


def plot_cellbin_gradient(
    adata: AnnData,
    mask: str,
    tag: str,
    prefix: str,
    colors: str = "RdYlBu_r",
    dpi: float = 600,
    edge_cut: float = 300,
    add_scale_bar: bool = True,
    add_legend=True,
    background: Literal["white", "black"] = "white",
    scale: float = 0.5,
    length_fraction: float = 0.25,
    ax: Optional[Axes] = None,
    save: bool = True,
) -> Axes:
    """Plot cellbin gradient image

    Args:
        adata (AnnData): AnnData object
        mask (str): mask file path
        tag (str): tag name in adata.obs
        prefix (str): prefix of output file
        colors (str, optional): color palette, defaults to "RdYlBu_r"
        dpi (float, optional): dpi of output file, defaults to 600
        edge_cut (float, optional): pixels to retain in edge cutting,
            defaults to 300
        add_scale_bar (bool, optional): add scale bar or not, defaults
            to True
        background (Literal[&quot;white&quot;, &quot;black&quot;], optional):
            background color, defaults to "white"
        scale (float, optional): scale bar length, defaults to 0.5
        length_fraction (float, optional): scale bar length fraction,
            defaults to 0.25
        ax (Optional[Axes], optional): matplotlib axes, defaults to None
        save (bool, optional): save figure or not, defaults to True

    Returns
    -------
        Axes: matplotlib axes
    """
    res = pd.DataFrame(adata.obs, columns=["x", "y", tag], index=adata.obs.index)
    res = res.sort_values(by=tag)
    vmax = res[tag].quantile(0.95)  # top 5% largest value as vmax
    res[tag][res[tag] > vmax] = vmax
    vmin = res[tag].quantile(0.05)  # bottom 5% smallest value as vmin
    res[tag][res[tag] < vmin] = vmin

    clusters = res[tag].unique()
    # cluster_number = clusters.shape[0]
    cmap = plt.cm.get_cmap("RdYlBu_r")  # get color plette
    norm = mcolors.Normalize(vmin=clusters.min(), vmax=clusters.max())
    colors = {value: mcolors.rgb2hex(cmap(norm(value))) for value in clusters}  # get color for each cluster

    # plot cellbin gradient image
    im = cell_bin_plot(mask=mask, res=res, tag=tag, colors=colors)

    # cut black edge of im
    # non_black_coords = np.argwhere(im.sum(axis=2) > 0)
    # y1, x1 = non_black_coords.min(axis=0)
    # y2, x2 = non_black_coords.max(axis=0)
    # im = im[y1 - edge_cut : y2 + edge_cut, x1 - edge_cut : x2 + edge_cut]
    im = crop_to_square_with_padding(im, edge_cut=edge_cut)

    # replace black background with white
    if background == "white":
        text_color = "black"
        black_pixels = (im[:, :, 0] == 0) & (im[:, :, 1] == 0) & (im[:, :, 2] == 0)
        im[black_pixels] = [255, 255, 255]
    else:
        text_color = "white"

    # draw colorbar
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi, gridspec_kw={"wspace": 0, "hspace": 0})
    #     plt.subplots_adjust(0, 0, 1, 1)
    if save:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi, gridspec_kw={"wspace": 0, "hspace": 0})
        fig.patch.set_facecolor(background)
        plt.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(im)

    # add scale bar
    if add_scale_bar:
        scalebar = ScaleBar(
            scale,
            "um",
            length_fraction=length_fraction,
            frameon=False,
            color=text_color,
            location="lower right",
        )
        ax.add_artist(scalebar)

    # Create colorbar and add to ax
    if add_legend:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
        cb.solids.set_edgecolor("face")
        cb.solids.set_facecolor(background)

        # 设置colorbar的字体颜色为白色
        cb.ax.yaxis.set_tick_params(color=text_color)
        cb.ax.xaxis.set_tick_params(color=text_color)
        cb.ax.tick_params(axis="x", colors=text_color, labelsize=14)
        cb.ax.tick_params(axis="y", colors=text_color, labelsize=14)
    if save:
        outfig = f"{tag}_{prefix}_spatial.pdf"
        plt.savefig(outfig, dpi=dpi, format="pdf", bbox_inches="tight")
    return im


def plot_cellbin_discrete(
    adata: AnnData,
    mask: str,
    tag: str,
    prefix: str,
    colors: Optional[Union[int, dict[str, str], str]] = 9,
    dpi: int = 600,
    edge_cut: int = 300,
    background: Literal["white", "black"] = "white",
    add_scale_bar: bool = True,
    scale: float = 0.5,
    length_fraction: float = 0.25,
    ax: Optional[Axes] = None,
    add_legend=True,
    save: bool = True,
) -> Axes:
    """Plot discrete cellbin images.

    Args:
        adata (AnnData): AnnData object
        mask (str): mask file path
        tag (str): tag name in adata.obs
        prefix (str): prefix of output file
        colors (Literal, optional): color palette in getDefaultcolors,
            defaults to 9
        dpi (int, optional): dpi of output file, defaults to 600
        edge_cut (int, optional): pixels to retain in edge cutting,
            defaults to 300
        background (Literal[&quot;white&quot;, &quot;black&quot;], optional):
            background color, white or black, defaults to "white"
        add_scale_bar (bool, optional): add scale bar or not, defaults
            to True
        scale (float, optional): scale bar length, defaults to 0.5
        length_fraction (float, optional): scale bar length fraction,
            defaults to 0.25
        ax (Optional[Axes], optional): matplotlib axes, defaults to None
        add_legend (bool, optional): add legend or not, defaults to True
        save (bool, optional): save figure or not, defaults to True

    Returns
    -------
        Axes: matplotlib axes
    """
    res = pd.DataFrame(adata.obs, columns=["x", "y", tag], index=adata.obs.index)
    res = res.sort_values(by=tag)
    # res.to_csv(f"bin1clu_{tag}_{prefix}.txt", sep="\t", index=False)  # write to file
    clusters = res[tag].unique()
    cluster_number = clusters.shape[0]
    if isinstance(colors, int) or isinstance(colors, str):
        colors = getDefaultColors(cluster_number, type=colors)
        im = cell_bin_plot(mask=mask, res=res, tag=tag, colors=colors)
    elif isinstance(colors, dict):
        im = cell_bin_plot(mask=mask, res=res, tag=tag, colors=colors)
    else:
        raise TypeError("Invalid type for 'colors'. Must be an int, str or dict.")

    # cut black edge of im
    # non_black_coords = np.argwhere(im.sum(axis=2) > 0)
    # y1, x1 = non_black_coords.min(axis=0)
    # y2, x2 = non_black_coords.max(axis=0)
    # im = im[y1 - edge_cut : y2 + edge_cut, x1 - edge_cut : x2 + edge_cut]
    im = crop_to_square_with_padding(im, edge_cut=edge_cut)

    # replace black background with white
    if background == "white":
        text_color = "black"
        black_pixels = (im[:, :, 0] == 0) & (im[:, :, 1] == 0) & (im[:, :, 2] == 0)
        im[black_pixels] = [255, 255, 255]
    else:
        text_color = "white"

    # new fig for save
    if save:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi, gridspec_kw={"wspace": 0, "hspace": 0})
        fig.patch.set_facecolor(background)
        plt.subplots_adjust(0, 0, 1, 1)

    ax.axis("off")
    ax.imshow(im)

    # add scale bar
    if add_scale_bar:
        scalebar = ScaleBar(
            scale,
            "um",
            length_fraction=length_fraction,
            frameon=False,
            color=text_color,
            location="lower right",
        )
        ax.add_artist(scalebar)

    # Create a legend for the discrete colors
    if add_legend:
        legend_labels = [str(cluster) for cluster in clusters]  # Assuming clusters is a list of labels
        legend_patches = [mpl.patches.Patch(color=colors[i], label=legend_labels[i]) for i in range(cluster_number)]
        legend = ax.legend(handles=legend_patches, loc="lower left", fontsize=8, bbox_to_anchor=(1.05, 0))
        legend.set_title(tag, prop={"size": 8, "weight": "bold"})
        # set legend text color white
        legend.get_title().set_color(text_color)
        for text in legend.get_texts():
            text.set_color(text_color)

    # save figure
    if save:
        outfig = f"{tag}_{prefix}_spatial.pdf"
        plt.savefig(outfig, dpi=dpi, format="pdf", bbox_inches="tight")

    return im


def plot_zoom_cellbin(
    adata, mask, tag, prefix, zoom_position1, zoom_position2, colors=10, zoom_size=3000, height=5, edge_cut=300
):
    im = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    im = crop_to_square_with_padding(im, edge_cut=edge_cut)
    figsize = im.shape
    print(figsize)
    zoom_size = 3000
    # 定义放大区域的坐标 (x, y, width, height)
    region1 = zoom_position1 + (zoom_size, zoom_size)
    region2 = zoom_position2 + (zoom_size, zoom_size)
    # 计算图片长宽
    width_ratios = figsize[1] / (figsize[0] / (zoom_size * 2) * zoom_size)
    print(width_ratios)
    width = (figsize[1] / (figsize[0] / height)) + ((figsize[1] / (figsize[0] / height)) / width_ratios)
    print(width)

    # 创建一个图形和网格
    fig = plt.figure(figsize=(width, height), facecolor="black")
    gs = gridspec.GridSpec(2, 2, width_ratios=[width_ratios, 1], hspace=0, wspace=0)

    # 绘制原图
    ax1 = plt.subplot(gs[:, 0])
    image = plot_cellbin_discrete(
        adata,
        mask=mask,
        tag=tag,
        prefix=prefix,
        colors=colors,
        dpi=1000,
        background="black",
        add_scale_bar=True,
        ax=ax1,
        add_legend=False,
        save=False,
    )
    ax1.axis("off")

    # 提取放大区域
    zoomed_region1 = image[region1[1] : region1[1] + region1[3], region1[0] : region1[0] + region1[2]]
    zoomed_region2 = image[region2[1] : region2[1] + region2[3], region2[0] : region2[0] + region2[2]]

    # 添加边框
    rect = patches.Rectangle((0, 0), figsize[1], figsize[0], linewidth=2, edgecolor="white", facecolor="none")
    ax1.add_patch(rect)

    # 在原图上标注放大区域（白色框体）
    rect1 = patches.Rectangle(
        (region1[0], region1[1]), region1[2], region1[3], linewidth=2, edgecolor="white", facecolor="none"
    )
    rect2 = patches.Rectangle(
        (region2[0], region2[1]), region2[2], region2[3], linewidth=2, edgecolor="white", facecolor="none"
    )
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)

    # 添加标题
    text_box = AnchoredText(
        prefix, frameon=True, loc=2, pad=0.4, borderpad=0.1, prop={"backgroundcolor": "white", "fontsize": 20}
    )
    ax1.add_artist(text_box)

    # 在原图上添加序号
    ax1.text(
        region1[0] + 1500,
        region1[1] + 1500,
        "1",
        color="white",
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax1.text(
        region2[0] + 1500,
        region2[1] + 1500,
        "2",
        color="white",
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # 绘制放大的区域1
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(zoomed_region1)
    ax2.axis("off")

    # 添加边框
    rect = patches.Rectangle((0, 0), zoom_size, zoom_size, linewidth=2, edgecolor="white", facecolor="none")
    ax2.add_patch(rect)

    # 在放大区域1上添加标题
    text_box = AnchoredText(
        "1", frameon=True, loc=2, pad=0.4, borderpad=0.1, prop={"backgroundcolor": "white", "fontsize": 20}
    )
    ax2.add_artist(text_box)

    # 绘制放大的区域2
    ax3 = plt.subplot(gs[1, 1])
    ax3.imshow(zoomed_region2)
    ax3.axis("off")

    # 添加边框
    rect = patches.Rectangle((0, 0), zoom_size, zoom_size, linewidth=2, edgecolor="white", facecolor="none")
    ax3.add_patch(rect)

    # 在放大区域2上添加标题
    text_box = AnchoredText(
        "2", frameon=True, loc=2, pad=0.4, borderpad=0.1, prop={"backgroundcolor": "white", "fontsize": 20}
    )
    ax3.add_artist(text_box)
    # 调整布局，使图像拼接成一个矩形
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f"cellbin_plot_zoom_{prefix}.pdf", dpi=300, bbox_inches="tight")
