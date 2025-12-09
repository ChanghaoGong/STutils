from __future__ import annotations

from pathlib import Path
from typing import Literal

import anndata.utils
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy._utils import Empty, _empty


def read_BGI_mtx(
    path: Path | str,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    cache: bool = False,
    cache_compression: Literal["gzip", "lzf"] | None | Empty = _empty,
    gex_only: bool = True,
    prefix: str | None = None,
    velocity: bool = False,
) -> AnnData:
    """Read BGI mtx file and return AnnData

    Args:
        path (Path | str): Path to mtx file
        var_names (Literal[&quot;gene_symbols&quot;, &quot;gene_ids&quot;], optional): . Defaults to "gene_symbols".
        make_unique (bool, optional): _description_. Defaults to True.
        cache (bool, optional): _description_. Defaults to False.
        cache_compression (Literal[&quot;gzip&quot;, &quot;lzf&quot;] | None | Empty, optional): _description_. Defaults to _empty.
        gex_only (bool, optional): _description_. Defaults to True.
        prefix (str | None, optional): _description_. Defaults to None.
        velocity (bool, optional): _description_. Defaults to False.

    Returns
    -------
        AnnData: _description_
    """
    path = Path(path)
    prefix = "" if prefix is None else prefix
    is_legacy = (path / f"{prefix}genes.tsv").is_file()
    adata = _read_BGI_mtx(
        path,
        var_names=var_names,
        make_unique=make_unique,
        cache=cache,
        cache_compression=cache_compression,
        prefix=prefix,
        is_legacy=is_legacy,
        velocity=velocity,
    )
    if is_legacy or not gex_only:
        return adata
    # gex_rows = adata.var["feature_types"] == "Gene Expression"
    # return adata[:, gex_rows].copy()
    return adata


def _read_BGI_mtx(
    path: Path,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    cache: bool = False,
    cache_compression: Literal["gzip", "lzf"] | None | Empty = _empty,
    prefix: str = "",
    is_legacy: bool,
    velocity: bool,
) -> AnnData:
    """Read mex from output from Cell Ranger v2- or v3+."""
    suffix = "" if is_legacy else ".gz"
    adata = sc.read(
        path / f"{prefix}matrix.mtx{suffix}",
        cache=cache,
        cache_compression=cache_compression,
    ).T  # transpose the data
    if velocity:
        adata.layers["spliced"] = sc.read(
            path.parent
            / "attachment"
            / "RNAvelocity_matrix"
            / f"{prefix}spliced.mtx{suffix}",
            cache=cache,
            cache_compression=cache_compression,
        ).T.X
        adata.layers["unspliced"] = sc.read(
            path.parent
            / "attachment"
            / "RNAvelocity_matrix"
            / f"{prefix}unspliced.mtx{suffix}",
            cache=cache,
            cache_compression=cache_compression,
        ).T.X
    genes = pd.read_csv(
        path / f"{prefix}{'genes' if is_legacy else 'features'}.tsv{suffix}",
        header=None,
        sep="\t",
    )
    if var_names == "gene_symbols":
        var_names_idx = pd.Index(genes[1].values)
        if make_unique:
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
        adata.var_names = var_names_idx
        adata.var["gene_ids"] = genes[0].values
    elif var_names == "gene_ids":
        adata.var_names = genes[0].values
        adata.var["gene_symbols"] = genes[1].values
    elif var_names == "BGI":
        var_names_idx = pd.Index(genes[0].values)
        if make_unique:
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
        adata.var_names = var_names_idx
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    # if not is_legacy:
    #    adata.var["feature_types"] = genes[2].values
    barcodes = pd.read_csv(path / f"{prefix}barcodes.tsv{suffix}", header=None)
    adata.obs_names = barcodes[0].values
    return adata
