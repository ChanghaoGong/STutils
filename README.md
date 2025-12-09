# STutils

[![Tests][badge-tests]][link-tests]
[![Build][badge-build]][link-build]
[![Documentation][badge-docs]][link-docs]
[![PyPI](https://img.shields.io/pypi/v/StereoUtils.svg)](https://pypi.org/project/StereoUtils)
[![Stars](https://img.shields.io/github/stars/ChanghaoGong/STutils?logo=GitHub)](https://github.com/ChanghaoGong/STutils/stargazers)

[badge-tests]: https://github.com/ChanghaoGong/STutils/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/ChanghaoGong/STutils/actions/workflows/test.yml
[badge-build]: https://github.com/ChanghaoGong/STutils/actions/workflows/build.yaml/badge.svg
[link-build]: https://github.com/ChanghaoGong/STutils/actions/workflows/build.yml
[badge-docs]: https://readthedocs.org/projects/stutils/badge/?version=latest

**STutils** is a Python package providing extra functions for analyzing STOmics (Spatial Transcriptomics) data, built on top of [scanpy](https://scanpy.readthedocs.io/) and [anndata](https://anndata.readthedocs.io/).

## Features

STutils provides three main modules for spatial transcriptomics analysis:

- **`pp` (Preprocessing)**: Data preprocessing and I/O functions
  - Read Stereo-seq formatted datasets
  - Merge cellbin data to metacells
  - Data format conversion

- **`tl` (Tools)**: Analysis tools and algorithms
  - Fast UCell pathway scoring
  - Fast gene signature scoring
  - Neighborhood enrichment analysis
  - Differential expression analysis
  - Cell density calculation
  - Bias gene removal

- **`pl` (Plotting)**: Visualization functions
  - Cellbin spatial plots (discrete and gradient)
  - Neighborhood heatmaps
  - Custom color palettes

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

### Install from GitHub

Install the latest development version:

```bash
pip install git+https://github.com/ChanghaoGong/STutils.git@main
```

### Dependencies

STutils requires:
- Python >= 3.10
- anndata
- scanpy
- session-info2

Optional dependencies for development, testing, and documentation are available. See `pyproject.toml` for details.

## Quick Start

### Reading Stereo-seq Data

```python
import STutils

# Read Stereo-seq formatted dataset
sdata = STutils.pp.stereoseq(
    path="path/to/stereo-seq/data", read_square_bin=True, optional_tif=False
)
```

### Spatial Visualization

```python
import STutils.pl as stpl

# Plot discrete cellbin image
stpl.plot_cellbin_discrete(
    adata=adata,
    mask="path/to/mask.tif",
    tag="cell_type",
    prefix="sample1",
    colors=10,
    save=True,
)

# Plot gradient cellbin image
stpl.plot_cellbin_gradient(
    adata=adata,
    mask="path/to/mask.tif",
    tag="gene_expression",
    prefix="sample1",
    colors="RdYlBu_r",
    save=True,
)
```

### Fast UCell Pathway Scoring

```python
import STutils.tl as sttl

# Step 1: Generate rank matrix
rank_matrix = sttl.fast_ucell_rank(
    adata=adata, n_cores_rank=4, maxRank=1500, rank_batch_size=100000
)

# Step 2: Calculate UCell scores
pathway_input = sttl.generate_pathway_input(adata=adata, pathway_dict=pathway_dict)

ucell_scores = sttl.fast_ucell_score(
    cell_index=list(adata.obs.index),
    rankmatrix=rank_matrix,
    n_cores_score=4,
    input_dict=pathway_input,
    maxRank=1500,
)
```

### Neighborhood Enrichment Analysis

```python
import STutils.tl as sttl

# Calculate neighborhood enrichment
nhood_percents = sttl.nhood_enrichment(
    adata=adata,
    coord_type="generic",
    library_key="batch",
    radius=30,
    cluster_key="cell_type",
)

# Plot neighborhood heatmap
import STutils.pl as stpl

stpl.nhood_heatmap(adata=adata, cluster_key="cell_type", radius=30, save=True)
```

### Differential Expression Analysis

```python
import scanpy as sc
import STutils.tl as sttl

# First, run rank_genes_groups
sc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon")

# Get DEGs with volcano plot
deg_dict = sttl.getDEG(
    adata=adata,
    cluster="cell_type",
    qval_cutoff=0.1,
    mean_expr_cutoff=0.2,
    top_genes=200,
    save="volcano_plot.pdf",
)
```

### Cell Density Calculation

```python
import STutils.tl as sttl

# Calculate cell type density
sttl.calculate_celltype_density(
    adata=adata,
    celltype_of_interest="T_cell",
    obs_key="cell_type",
    coord_keys=["x", "y"],
    plot_result=True,
)

# Calculate co-culture probability
sttl.calculate_coculture_probability(
    adata=adata,
    celltype1="T_cell",
    celltype2="B_cell",
    obs_key="cell_type",
    coord_keys=["x", "y"],
)
```

## Main Modules

### Preprocessing (`pp`)

- **`stereoseq()`**: Read Stereo-seq formatted datasets into SpatialData objects
- **`merge_big_cell()`**: Merge STomics cellbin data to metacells by axis and celltype

### Tools (`tl`)

- **`fast_ucell_rank()`**: Fast ranking step for UCell pathway scoring
- **`fast_ucell_score()`**: Fast UCell pathway scoring
- **`fast_sctl_score()`**: Fast multi-core version of scanpy.tl.score_genes()
- **`nhood_enrichment()`**: Calculate neighborhood enrichment between cell types
- **`test_nhood()`**: Analyze cell type neighborhood interactions with spatial shuffling test
- **`getDEG()`**: Extract differentially expressed genes with volcano plots
- **`calculate_celltype_density()`**: Calculate spatial density distribution of cell types
- **`calculate_coculture_probability()`**: Calculate co-occurrence probability of two cell types
- **`generate_pathway_input()`**: Prepare pathway input dictionary for scoring functions
- **`removeBiasGenes()`**: Remove bias genes from analysis

### Plotting (`pl`)

- **`plot_cellbin_discrete()`**: Plot discrete cellbin images with categorical colors
- **`plot_cellbin_gradient()`**: Plot gradient cellbin images with continuous color scales
- **`plot_zoom_cellbin()`**: Plot zoomed-in regions of cellbin images
- **`nhood_heatmap()`**: Plot neighborhood enrichment heatmaps
- **`getDefaultColors()`**: Get beautiful color palettes for scientific plotting

## Documentation

Please refer to the [documentation][link-docs] for detailed API reference and tutorials. In particular:

- [API documentation][link-api]
- [Changelog][changelog]

## Citation

If you use STutils in your research, please cite:

> t.b.a

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## License

See LICENSE file for details.

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/ChanghaoGong/STutils/issues
[changelog]: https://stutils.readthedocs.io/en/latest/changelog.html
[link-docs]: https://stutils.readthedocs.io/en/latest/?badge=latest
[link-api]: https://stutils.readthedocs.io/en/latest/api.html
