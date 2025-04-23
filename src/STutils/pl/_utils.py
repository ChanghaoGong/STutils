import random
from collections.abc import Mapping
from typing import Literal, Union

import cv2
import matplotlib
import matplotlib.cm as cm
import numpy as np


def hex_to_rgb(hex_color) -> tuple[int, int, int]:
    """Convert hex color code to rgb

    params:
        hex_color: hex color code, like '#FFFFFF'
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def int_to_rgb_idx(
    arr: np.ndarray,
    mapping: Mapping[int, tuple[int, int, int]],
    default: tuple = (0, 0, 0),
) -> np.ndarray:
    """Convert int to rgb index by color mapping dict

    Args:
        arr (np.ndarray): result matrix of connectedComponentsWithStats
        mapping (Mapping[int, Tuple[int, int, int]]): mapping dict
        default (Tuple, optional): background color. Defaults to (0, 0, 0).

    Returns
    -------
        np.ndarray: image array
    """
    mapping = {k: mapping.get(k, default) for k in range(int(arr.max()) + 1)}
    colormap = np.array([*mapping.values()])
    return colormap[arr]


def crop_to_square_with_padding(im, edge_cut=300):
    """
    裁剪图像，使其包含所有非纯黑色像素，并扩展为正方形。较短的一侧会用黑色填充以使长宽相等。

    参数:
    - im: str，图像。
    - edge_cut: int，扩展边缘的像素数，默认为300。

    返回:
    - im_final: np.ndarray，裁剪并扩展后的正方形图像。
    """
    if len(im.shape) == 2:
        gray = im
    else:
        # 将RGB图像转换为灰度图像
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 定义纯黑色的阈值
    threshold = 10  # 可以根据需要调整这个值

    # 创建掩码
    mask = gray > threshold

    # 找到非纯黑色像素的坐标
    non_black_coords = np.argwhere(mask)
    y1, x1 = non_black_coords.min(axis=0)
    y2, x2 = non_black_coords.max(axis=0)

    # 裁剪非纯黑色像素的区域
    cropped_im = im[y1:y2, x1:x2]

    # 如果裁剪后的图像只有一个通道，则添加一个通道维度
    if len(cropped_im.shape) == 2:
        cropped_im = np.expand_dims(cropped_im, axis=-1)

    # 计算裁剪区域的长和宽
    height, width, _ = cropped_im.shape

    # 选择较大的维度
    max_dim = max(height, width) + edge_cut // 2
    # 创建一个新的正方形图像，用黑色填充
    im_square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

    # 将裁剪后的图像放在新图像的中心
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2
    im_square[y_offset : y_offset + height, x_offset : x_offset + width] = cropped_im

    return im_square


def getDefaultColors(n: int, type: Union[Literal[1], list] = 1) -> list:
    """A beautiful color series list for sci plotting!

    Args:
        n (int): number of colors to generate
        type (Literal, optional): type, defaults to 1

    Returns
    -------
        list: a color rgb hex list
    """
    if type == 1:
        colors = [
            "#ff1a1a",
            "#1aff1a",
            "#1a1aff",
            "#ffff1a",
            "#ff1aff",
            "#ff8d1a",
            "#7cd5c8",
            "#c49a3f",
            "#5d8d9c",
            "#90353b",
            "#507d41",
            "#502e71",
            "#1B9E77",
            "#c5383c",
            "#0081d1",
            "#674c2a",
            "#c8b693",
            "#aed688",
            "#f6a97a",
            "#c6a5cc",
            "#798234",
            "#6b42c8",
            "#cf4c8b",
            "#666666",
            "#ffd900",
            "#feb308",
            "#cb7c77",
            "#68d359",
            "#6a7dc9",
            "#c9d73d",
        ]
    elif type == 2:
        if n <= 14:
            colors = [
                "#437BFE",
                "#FEC643",
                "#43FE69",
                "#FE6943",
                "#C643FE",
                "#43D9FE",
                "#B87A3D",
                "#679966",
                "#993333",
                "#7F6699",
                "#E78AC3",
                "#333399",
                "#A6D854",
                "#E5C494",
            ]
        elif n <= 20:
            colors = [
                "#87b3d4",
                "#d5492f",
                "#6bd155",
                "#683ec2",
                "#c9d754",
                "#d04dc7",
                "#81d8ae",
                "#d34a76",
                "#607d3a",
                "#6d76cb",
                "#ce9d3f",
                "#81357a",
                "#d3c3a4",
                "#3c2f5a",
                "#b96f49",
                "#4e857e",
                "#6e282c",
                "#d293c8",
                "#393a2a",
                "#997579",
            ]
        elif n <= 30:
            colors = [
                "#628bac",
                "#ceda3f",
                "#7e39c9",
                "#72d852",
                "#d849cc",
                "#5e8f37",
                "#5956c8",
                "#cfa53f",
                "#392766",
                "#c7da8b",
                "#8d378c",
                "#68d9a3",
                "#dd3e34",
                "#8ed4d5",
                "#d84787",
                "#498770",
                "#c581d3",
                "#d27333",
                "#6680cb",
                "#83662e",
                "#cab7da",
                "#364627",
                "#d16263",
                "#2d384d",
                "#e0b495",
                "#4b272a",
                "#919071",
                "#7b3860",
                "#843028",
                "#bb7d91",
            ]
        else:
            colors = [
                "#982f29",
                "#5ddb53",
                "#8b35d6",
                "#a9e047",
                "#4836be",
                "#e0dc33",
                "#d248d5",
                "#61a338",
                "#9765e5",
                "#69df96",
                "#7f3095",
                "#d0d56a",
                "#371c6b",
                "#cfa738",
                "#5066d1",
                "#e08930",
                "#6a8bd3",
                "#da4f1e",
                "#83e6d6",
                "#df4341",
                "#6ebad4",
                "#e34c75",
                "#50975f",
                "#d548a4",
                "#badb97",
                "#b377cf",
                "#899140",
                "#564d8b",
                "#ddb67f",
                "#292344",
                "#d0cdb8",
                "#421b28",
                "#5eae99",
                "#a03259",
                "#406024",
                "#e598d7",
                "#343b20",
                "#bbb5d9",
                "#975223",
                "#576e8b",
                "#d97f5e",
                "#253e44",
                "#de959b",
                "#417265",
                "#712b5b",
                "#8c6d30",
                "#a56c95",
                "#5f3121",
                "#8f846e",
                "#8f5b5c",
            ]
    elif type == 3:
        colors = [
            "#588dd5",
            "#c05050",
            "#07a2a4",
            "#f5994e",
            "#9a7fd1",
            "#59678c",
            "#c9ab00",
            "#7eb00a",
        ]
    elif type == 4:
        colors = [
            "#FC8D62",
            "#66C2A5",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]
    elif type == 5:
        colors = [
            "#c14089",
            "#6f5553",
            "#E5C494",
            "#738f4c",
            "#bb6240",
            "#66C2A5",
            "#2dfd29",
            "#0c0fdc",
        ]
    elif type == 6:
        colors = [
            "#F9423A",
            "#C2A91A",
            "#486AFF",
            "#FA8155",
            "#92A022",
            "#7574BC",
            "#E7C2C0",
            "#D8DCAF",
            "#B0C0D0",
            "#D3BD6C",
            "#76B18A",
            "#5192A4",
            "#F2D96E",
            "#C53F4D",
            "#0B5D99",
            "#7B4083",
        ]
    elif type == 7:
        colors = [
            "#C77CFF",
            "#7CAE00",
            "#00BFC4",
            "#F8766D",
            "#AB82FF",
            "#90EE90",
            "#00CD00",
            "#008B8B",
            "#FFA500",
        ]
    elif type == 8:
        colors = [
            "#9af764",
            "#3e82fc",
            "#fe0002",
            "#f4d054",
            "#ed0dd9",
            "#13eac9",
            "#e4cbff",
            "#b1d27b",
            "#ad8150",
            "#601ef9",
            "#ff9408",
            "#75bbfd",
            "#fdb0c0",
            "#a50055",
            "#4da409",
            "#c04e01",
            "#d2bd0a",
            "#ada587",
            "#0504aa",
            "#650021",
            "#d0fefe",
            "#a8ff04",
            "#fe46a5",
            "#bc13fe",
            "#fdff52",
            "#f2ab15",
            "#fd4659",
            "#ff724c",
            "#cba560",
            "#cbf85f",
            "#78d1b6",
            "#9d0216",
            "#874c62",
            "#8b88f8",
            "#05472a",
            "#b17261",
            "#a4be5c",
            "#742802",
            "#3e82fc",
            "#eedc5b",
            "#a8a495",
            "#fffe71",
            "#c1c6fc",
            "#b17261",
            "#ff5b00",
            "#f10c45",
            "#3e82fc",
            "#de9dac",
            "#f10c45",
            "#056eee",
            "#e6daa6",
            "#eedc5b",
            "#c87606",
            "#9dbcd4",
            "#56ae57",
            "#49759c",
            "#d8dcd6",
        ]
    elif type == 9:
        colors = [
            "#a2cffe",
            "#87a922",
            "#ffa62b",
            "#f8481c",
            "#cffdbc",
            "#a6814c",
            "#a484ac",
            "#fc86aa",
            "#952e8f",
            "#02ccfe",
            "#2000b1",
            "#009337",
            "#ad0afd",
            "#3c9992",
            "#d8dcd6",
            "#cb6843",
        ]
    elif type == 10:
        colors = [
            "#d77a7f",
            "#8eda48",
            "#7340cd",
            "#d6c847",
            "#ce4cc5",
            "#64db8e",
            "#432876",
            "#509140",
            "#7171cd",
            "#d1863a",
            "#79acd9",
            "#d24530",
            "#6dc7b7",
            "#d23e70",
            "#c6d394",
            "#8d3870",
            "#827f38",
            "#cd90cb",
            "#3a4e32",
            "#c9c5c6",
            "#3e263b",
            "#ae8875",
            "#556983",
            "#753627",
        ]
    elif type == "rdgy":
        colors = ["#F9423A", "#c9c5c6"]
    elif type in [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ]:
        cmap = cm.get_cmap(type)
        colors = [matplotlib.colors.rgb2hex(rgba) for rgba in cmap.colors]
    # if "type" is a list retrun itself
    elif isinstance(type, list):
        colors = type

    if n:
        if n <= len(colors):
            colors = colors[:n]
        else:
            step = 16777200 // (n - len(colors)) - 2
            add_colors = []
            tmp = random.sample(range(step), 1)[0]
            for i in range(n - len(colors)):
                print(i)
                hextmp = f"{tmp:06X}"
                add_colors.append("#" + hextmp)
                tmp = tmp + step
            colors = colors + add_colors
    return colors
