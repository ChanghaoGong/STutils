import numpy as np
import scanpy as sc
from scipy.stats import gaussian_kde


def calculate_celltype_density(
    adata,
    celltype_of_interest,
    obs_key="celltype",
    coord_keys: list[str] | None = None,
    scotts_factor_scale=10,
    density_colname=None,
    plot_result=True,
    plot_kwargs=None,
):
    """
    Calculate the density distribution of a specific cell type using Gaussian KDE.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates and cell type information.
    celltype_of_interest : str
        The cell type for which to calculate density.
    obs_key : str, optional (default: 'celltype')
        Key in adata.obs where cell type information is stored.
    coord_keys : list of str, optional (default: ['x', 'y'])
        Keys in adata.obs for spatial coordinates.
    scotts_factor_scale : float, optional (default: 10)
        Scaling factor for the Scott's rule bandwidth in KDE.
    density_colname : str, optional
        Name of the column to store density values in adata.obs.
        If None, will use f"{celltype_of_interest}_density".
    plot_result : bool, optional (default: True)
        Whether to plot the density results.
    plot_kwargs : dict, optional
        Additional arguments to pass to sc.pl.spatial.

    Returns
    -------
    adata : AnnData
        Updated AnnData object with density values added to obs.
    """
    # Set default column name for density if not provided
    if density_colname is None:
        density_colname = f"{celltype_of_interest.lower()}_density"

    # Extract coordinates and cell types
    coords = adata.obs[coord_keys].values
    cell_types = adata.obs[obs_key].values

    # Filter coordinates for the cell type of interest
    target_coords = coords[cell_types == celltype_of_interest]

    # Calculate KDE if there are cells of this type
    if len(target_coords) > 0:
        # Compute Scott's factor and apply scaling
        scotts_factor = np.power(len(target_coords), -1 / (2 + 4))
        kde = gaussian_kde(
            target_coords.T, bw_method=scotts_factor / scotts_factor_scale
        )

        # Evaluate density at all points
        densities = kde(coords.T)
    else:
        print(f"Warning: No cells of type '{celltype_of_interest}' found.")
        densities = np.zeros(len(coords))

    # Normalize densities to 0-1 range
    normalized_densities = (densities - densities.min()) / (
        densities.max() - densities.min() + 1e-10
    )

    # Add to adata.obs
    adata.obs[density_colname] = normalized_densities

    # Plot results if requested
    if plot_result:
        default_plot_kwargs = {
            "vmax": "p99",
            "vmin": "p1",
            "spot_size": 30,
            "color_map": "Spectral_r",
            "title": f"{celltype_of_interest} density",
        }

        if plot_kwargs is not None:
            default_plot_kwargs.update(plot_kwargs)

        sc.pl.spatial(adata, color=density_colname, **default_plot_kwargs)

    return density_colname


def calculate_coculture_probability(
    adata,
    celltype1,
    celltype2,
    obs_key="celltype",
    coord_keys: list[str] | None = None,
    scotts_factor_scale=10,
    colname="coculture_probability",
    plot_result=True,
    plot_kwargs=None,
):
    """
    Calculate the co-occurrence probability of two cell types in space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    celltype1, celltype2 : str
        Names of the two cell types to analyze.
    obs_key : str (default: 'celltype')
        Key in adata.obs for cell type labels.
    coord_keys : list (default: ['x', 'y'])
        Keys in adata.obs for spatial coordinates.
    scotts_factor_scale : float (default: 10)
        Bandwidth scaling factor for KDE.
    colname : str (default: 'coculture_probability')
        Column name to store results in adata.obs.
    plot_result : bool (default: True)
        Whether to plot the results.
    plot_kwargs : dict (optional)
        Additional arguments for sc.pl.spatial.

    Returns
    -------
    adata : Updated AnnData with co-occurrence probabilities.
    """
    if coord_keys is None:
        coord_keys = ["x", "y"]
    # Extract coordinates and cell types
    coords = adata.obs[coord_keys].values
    cell_types = adata.obs[obs_key].values

    # Function to calculate density for a single cell type
    def _get_density(target_celltype):
        target_coords = coords[cell_types == target_celltype]
        if len(target_coords) > 0:
            scotts_factor = np.power(len(target_coords), -1 / (2 + 4))
            kde = gaussian_kde(
                target_coords.T, bw_method=scotts_factor / scotts_factor_scale
            )
            density = kde(coords.T)
            return (density - density.min()) / (density.max() - density.min() + 1e-10)
        else:
            print(f"Warning: No cells of type '{target_celltype}' found.")
            return np.zeros(len(coords))

    # Calculate densities for both cell types
    density1 = _get_density(celltype1)
    density2 = _get_density(celltype2)

    # Calculate joint probability (normalized product)
    joint_prob = density1 * density2
    joint_prob = joint_prob / (joint_prob.max() + 1e-10)  # Normalize to [0,1]

    # Store results
    adata.obs[colname] = joint_prob

    # Plot if requested
    if plot_result:
        default_kwargs = {
            "vmax": "p99",
            "vmin": 0,
            "spot_size": 30,
            "color_map": "viridis",
            "title": f"{celltype1}-{celltype2} co-occurrence",
        }
        if plot_kwargs:
            default_kwargs.update(plot_kwargs)

        sc.pl.spatial(adata, color=colname, **default_kwargs)

    return adata
