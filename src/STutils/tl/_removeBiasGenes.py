import numpy as np
from anndata import AnnData


def removeBiasGenes(adata: AnnData) -> AnnData:
    """Remove unused genes for human singcle cell analysis

    :param adata: Anndata object
    :type adata: AnnData
    :return: filtered anndata
    :rtype: AnnData
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
        malat1 | MTgenes | hb_genes | RPgenes | RPgenes2 | CTCgenes | MIRgenes | ACgenes | CTgenes | LINCgenes | ALgenes
    )
    keep = np.invert(remove_genes)
    adata = adata[:, keep]
