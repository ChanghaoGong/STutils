from collections.abc import Sequence
from ctypes import c_uint32
from math import ceil
from multiprocessing import Array, Process, cpu_count
from multiprocessing.sharedctypes import RawArray
from operator import attrgetter, mul

import numpy as np
import pandas as pd
from anndata import AnnData
from boltons.iterutils import chunked
from ctxcore.genesig import GeneSignature
from ctxcore.recovery import enrichment4cells
from scipy.sparse import csr_matrix, issparse, spmatrix
from tqdm import tqdm

DTYPE = "uint32"
DTYPE_C = c_uint32


def create_rankings(ex_mtx: pd.DataFrame | spmatrix, seed=None) -> pd.DataFrame:
    """Create a whole genome rankings dataframe from a single cell expression profile dataframe.

    Args:
        ex_mtx: The expression profile matrix. The rows should
            correspond to different cells, the columns to different
            genes (n_cells x n_genes). Can be a pandas DataFrame or sparse matrix.
        seed: Random seed for reproducibility. If None, a random seed will be used.
            Used for shuffling columns to handle ties randomly.

    Returns
    -------
        A genome rankings dataframe (n_cells x n_genes).
    """
    # Handle sparse matrix efficiently
    if issparse(ex_mtx):
        # Convert to CSR format for efficient row operations
        if not isinstance(ex_mtx, csr_matrix):
            ex_mtx = csr_matrix(ex_mtx)

        n_cells, n_genes = ex_mtx.shape

        # Initialize ranking matrix
        rankings = np.zeros((n_cells, n_genes), dtype=DTYPE)

        # Set up random number generator for reproducibility
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Process each row (cell) efficiently using CSR format
        # CSR format: data, indices, indptr
        # Access row i: data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]]
        for i in range(n_cells):
            # Get row start and end indices in CSR format
            row_start = ex_mtx.indptr[i]
            row_end = ex_mtx.indptr[i + 1]

            # Optimized sparse ranking: work directly with non-zero values
            if row_end > row_start:
                # Row has non-zero values
                # Initialize row data with zeros (sparse representation)
                # For sparse matrices, most values are zero, so we only fill non-zeros
                row_data = np.zeros(n_genes, dtype=ex_mtx.dtype)

                # Fill non-zero values from sparse matrix using direct indexing
                # This avoids creating full dense arrays unnecessarily
                row_indices = ex_mtx.indices[row_start:row_end]
                row_values = ex_mtx.data[row_start:row_end]
                row_data[row_indices] = row_values

                # Shuffle column indices to handle ties randomly
                # Each row gets its own random permutation for proper tie-breaking
                shuffled_cols = rng.permutation(n_genes)
                row_data_shuffled = row_data[shuffled_cols]

                # Calculate ranks: descending order (high expression = low rank)
                # Use argsort with 'stable' method for tie-breaking (equivalent to 'first')
                # This maintains the order of equal values, which is important for tie-breaking
                sort_indices = np.argsort(-row_data_shuffled, kind="stable")

                # Assign ranks: 0-based (subtract 1 from 1-based rank)
                ranks = np.zeros(n_genes, dtype=DTYPE)
                ranks[sort_indices] = np.arange(n_genes, dtype=DTYPE)

                # Handle NaN values (should get highest rank)
                nan_mask = np.isnan(row_data_shuffled)
                if nan_mask.any():
                    # NaN values get the highest ranks
                    max_rank = n_genes - nan_mask.sum()
                    ranks[nan_mask] = np.arange(max_rank, n_genes, dtype=DTYPE)

                # Map back to original column order using inverse permutation
                # shuffled_cols maps original -> shuffled, we need shuffled -> original
                inv_shuffled = np.argsort(shuffled_cols)
                rankings[i] = ranks[inv_shuffled]
            else:
                # Entire row is zero - optimize: all genes have same expression (0)
                # Shuffle for random tie-breaking, then assign ranks
                shuffled_cols = rng.permutation(n_genes)
                # For all-zero row, ranks are just the shuffled order
                ranks = np.zeros(n_genes, dtype=DTYPE)
                ranks[shuffled_cols] = np.arange(n_genes, dtype=DTYPE)
                # Map back to original column order
                inv_shuffled = np.argsort(shuffled_cols)
                rankings[i] = ranks[inv_shuffled]

        # Create DataFrame with original indices and columns
        # Note: sparse matrices don't have index/columns, so we use RangeIndex
        index = pd.RangeIndex(n_cells)
        columns = pd.RangeIndex(n_genes)

        return pd.DataFrame(rankings, index=index, columns=columns)

    # Original dense matrix implementation
    # Do a shuffle would be nice for exactly similar behaviour as R implementation.
    # 1. Ranks are assigned in the range of 1 to n, therefore we need to subtract 1.
    # 2. In case of a tie the 'first' method is used, i.e. we keep the order in the original array. The remove any
    #    bias we shuffle the dataframe before ranking it. This introduces a performance penalty!
    # 3. Genes are ranked according to gene expression in descending order, i.e. from highly expressed (0) to low expression (n).
    # 3. NAs should be given the highest rank numbers. Documentation is bad, so tested implementation via code snippet:
    #
    #    import pandas as pd
    #    import numpy as np
    #    df = pd.DataFrame(data=[4, 1, 3, np.nan, 2, 3], columns=['values'])
    #    # Run below statement multiple times to see effect of shuffling in case of a tie.
    #    df.sample(frac=1.0, replace=False).rank(ascending=False, method='first', na_option='bottom').sort_index() - 1
    #
    return (
        ex_mtx.sample(frac=1.0, replace=False, axis=1, random_state=seed)
        .rank(axis=1, ascending=False, method="first", na_option="bottom")
        .astype(DTYPE)
        - 1
    )


def derive_auc_threshold(ex_mtx: pd.DataFrame | spmatrix) -> pd.DataFrame:
    """Derive AUC thresholds for an expression matrix.

    It is important to check that most cells have a substantial fraction of expressed/detected genes in the calculation of
    the AUC.

    Args:
        ex_mtx: The expression profile matrix. The rows should
            correspond to different cells, the columns to different
            genes (n_cells x n_genes). Can be a pandas DataFrame or sparse matrix.

    Returns
    -------
        A dataframe with AUC threshold for different quantiles over the
        number cells: a fraction of 0.01 designates that when using this
        value as the AUC threshold for 99% of the cells all ranked genes
        used for AUC calculation will have had a detected expression in
        the single-cell experiment.
    """
    # Handle sparse matrix efficiently
    if issparse(ex_mtx):
        # Use getnnz for efficient counting of non-zero elements per row
        non_zero_counts = np.array(ex_mtx.getnnz(axis=1)).flatten()
    else:
        non_zero_counts = np.count_nonzero(ex_mtx, axis=1)

    return (
        pd.Series(non_zero_counts).quantile([0.01, 0.05, 0.10, 0.50, 1])
        / ex_mtx.shape[1]
    )


enrichment = enrichment4cells


def _enrichment(
    shared_ro_memory_array, modules, genes, cells, auc_threshold, auc_mtx, offset
):
    # The rankings dataframe is properly reconstructed (checked this).
    df_rnk = pd.DataFrame(
        data=np.frombuffer(shared_ro_memory_array, dtype=DTYPE).reshape(
            len(cells), len(genes)
        ),
        columns=genes,
        index=cells,
    )
    # To avoid additional memory burden de resulting AUCs are immediately stored in the output sync. array.
    result_mtx = np.frombuffer(auc_mtx.get_obj(), dtype="d")
    inc = len(cells)
    for idx, module in enumerate(modules):
        result_mtx[offset + (idx * inc) : offset + ((idx + 1) * inc)] = (
            enrichment4cells(df_rnk, module, auc_threshold).values.ravel(order="C")
        )


def aucell4r(
    df_rnk: pd.DataFrame,
    signatures: Sequence[type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """Calculate enrichment of gene signatures for single cells.

    Args:
        df_rnk: The rank matrix (n_cells x n_genes).
        signatures: The gene signatures or regulons.
        auc_threshold: The fraction of the ranked genome to take into
            account for the calculation of the Area Under the recovery
            Curve.
        noweights: Should the weights of the genes part of a signature
            be used in calculation of enrichment?
        normalize: Normalize the AUC values to a maximum of 1.0 per
            regulon.
        num_workers: The number of cores to use.

    Returns
    -------
        A dataframe with the AUCs (n_cells x n_modules).
    """
    if num_workers == 1:
        # Show progress bar ...
        aucs = pd.concat(
            [
                enrichment4cells(
                    df_rnk,
                    module.noweights() if noweights else module,
                    auc_threshold=auc_threshold,
                )
                for module in tqdm(signatures)
            ]
        ).unstack("Regulon")
        aucs.columns = aucs.columns.droplevel(0)
    else:
        # Decompose the rankings dataframe: the index and columns are shared with the child processes via pickling.
        genes = df_rnk.columns.values
        cells = df_rnk.index.values
        # The actual rankings are shared directly. This is possible because during a fork from a parent process the child
        # process inherits the memory of the parent process. A RawArray is used instead of a synchronize Array because
        # these rankings are read-only.
        shared_ro_memory_array = RawArray(DTYPE_C, mul(*df_rnk.shape))
        array = np.frombuffer(shared_ro_memory_array, dtype=DTYPE)
        # Copy the contents of df_rank into this shared memory block using row-major ordering.
        array[:] = df_rnk.values.ravel(order="C")

        # The resulting AUCs are returned via a synchronize array.
        # Double precision floats.
        auc_mtx = Array("d", len(cells) * len(signatures))

        # Convert the modules to modules with uniform weights if necessary.
        if noweights:
            signatures = [m.noweights() for m in signatures]

        # Do the analysis in separate child processes.
        chunk_size = ceil(float(len(signatures)) / num_workers)
        processes = [
            Process(
                target=_enrichment,
                args=(
                    shared_ro_memory_array,
                    chunk,
                    genes,
                    cells,
                    auc_threshold,
                    auc_mtx,
                    (chunk_size * len(cells)) * idx,
                ),
            )
            for idx, chunk in enumerate(chunked(signatures, chunk_size))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Reconstitute the results array. Using C or row-major ordering.
        aucs = pd.DataFrame(
            data=np.ctypeslib.as_array(auc_mtx.get_obj()).reshape(
                len(signatures), len(cells)
            ),
            columns=pd.Index(data=cells, name="Cell"),
            index=pd.Index(
                data=list(map(attrgetter("name"), signatures)), name="Regulon"
            ),
        ).T
    return aucs / aucs.max(axis=0) if normalize else aucs


def aucell(
    adata: AnnData,
    signatures: Sequence[type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    seed: int = 42,
    num_workers: int = cpu_count(),
    use_raw: bool | None = None,
) -> list[str]:
    """Calculate enrichment of gene signatures for single cells.

    This function efficiently handles both dense and sparse expression matrices.
    For sparse matrices, it uses optimized CSR format operations to avoid
    unnecessary memory usage.

    Parameters
    ----------
    adata
        The annotated data matrix.
    signatures
        The gene signatures or regulons to calculate enrichment for.
    auc_threshold
        The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    noweights
        Should the weights of the genes part of a signature be used in calculation of enrichment?
    seed
        The random seed for sampling. Used for reproducibility in ranking calculations.
    num_workers
        The number of CPU cores to use for parallel processing.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.

    Returns
    -------
    list[str]
        A list of column names for the calculated AUC values, which are also added to `adata.obs`.
    """
    exp_mtx = adata.raw.X if use_raw else adata.X

    # Keep sparse format if possible - only convert when necessary
    # The create_rankings function now handles sparse matrices efficiently
    if issparse(exp_mtx):
        # Pass sparse matrix directly - create_rankings will handle it efficiently
        rankings = create_rankings(exp_mtx, seed)
        # Set proper index and columns from adata (sparse matrices don't have these)
        rankings.index = adata.obs_names
        rankings.columns = adata.var_names
    else:
        # Dense matrix: convert to DataFrame as before
        exp_mtx = pd.DataFrame(exp_mtx, index=adata.obs_names, columns=adata.var_names)
        rankings = create_rankings(exp_mtx, seed)

    auc = aucell4r(
        rankings,
        signatures,
        auc_threshold,
        noweights,
        False,
        num_workers,
    )
    auc.columns = "AUC_" + auc.columns
    adata.obs[auc.columns] = auc
    return auc.columns.tolist()
