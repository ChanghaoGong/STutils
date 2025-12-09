import numpy as np
from anndata import AnnData


def removeBiasGenes(adata: AnnData) -> AnnData:
    """Remove bias genes for human single cell analysis.

    This function removes common bias genes that are typically not informative
    for downstream analysis, including mitochondrial genes, ribosomal proteins,
    hemoglobin genes, and various non-coding RNAs.

    Args:
        adata: Annotated data object containing gene expression data.

    Returns
    -------
        A new AnnData object with bias genes removed. The original object is not modified.
    """
    malat1 = adata.var_names.str.startswith("MALAT1")
    MTgenes = adata.var_names.str.startswith("MT-")
    hb_genes = adata.var_names.str.contains("^HB[^(P)]")
    RPgenes = adata.var_names.str.startswith("RP") & adata.var_names.str.contains("-")
    RPgenes2 = adata.var_names.str.contains("^RP[SL]")
    CTCgenes = adata.var_names.str.startswith("CTC") & adata.var_names.str.contains("-")
    MIRgenes = adata.var_names.str.startswith("MIR")
    ACgenes = adata.var_names.str.contains("^AC[0-9]") & adata.var_names.str.contains(".")
    CTgenes = adata.var_names.str.startswith("CT") & adata.var_names.str.contains("-")
    LINCgenes = adata.var_names.str.contains("^LINC[0-9]")
    ALgenes = adata.var_names.str.contains("^AL") & adata.var_names.str.contains(".")
    remove_genes = (
        malat1
        | MTgenes
        | hb_genes
        | RPgenes
        | RPgenes2
        | CTCgenes
        | MIRgenes
        | ACgenes
        | CTgenes
        | LINCgenes
        | ALgenes
    )
    keep = np.invert(remove_genes)
    adata_filtered = adata[:, keep]
    return adata_filtered
