"""
Vegetation Masking and Forest Structure Index Module

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Processes canopy cover data to produce a tree cover percentage raster
    (0-100) for PRA fuzzy logic, a Forest Structure Index (FSI) raster
    (0-1) for the FlowPy forest module, and shrub cover percentage
    (LANDFIRE EVC input only). Supports two canopy source types via
    VegetationConfig.canopy_source_type: "direct_pct" (values are canopy
    percent, reprojected bilinear) and "evc_encoded" (LANDFIRE EVC
    categorical codes, reprojected nearest).
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import logging

from config import VegetationConfig

logger = logging.getLogger(__name__)


# --------- Source-Type Helpers ---------

def _get_source_type(config):
    # Return canopy source type, defaulting to direct_pct for configs
    # that predate the field.
    return getattr(config, "canopy_source_type", "direct_pct")


# --------- Canopy Loading ---------

def load_canopy(canopy_path):
    # Load canopy raster (EVC-encoded or direct-percent).
    # @param canopy_path: Path to canopy GeoTIFF
    # @returns: dict with keys: data, transform, crs, nodata, shape, dtype

    canopy_path = Path(canopy_path)

    with rasterio.open(canopy_path) as src:
        data = src.read(1)
        result = {
            'data': data,
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'shape': data.shape,
            'dtype': data.dtype,
        }

    logger.info(f"Loaded canopy raster: {canopy_path.name} ({data.shape}, "
                f"dtype={data.dtype}, nodata={src.nodata}, crs={src.crs})")
    return result


# --------- Cover Extraction ---------

def extract_tree_cover_percent(canopy, config=None):
    # Extract tree cover percentage (0-100) from canopy raster values.
    # direct_pct: values ARE the percent (clipped, nodata -> 0).
    # evc_encoded: decode LANDFIRE range 110-199 to 10-99%.
    # @param canopy: 2D numpy array of canopy raster values
    # @param config: VegetationConfig instance (uses defaults if None)
    # @returns: numpy array - tree cover percentage (0-100), float32

    if config is None:
        config = VegetationConfig()

    source_type = _get_source_type(config)

    if source_type == "direct_pct":
        tree_cover = canopy.astype(np.float32)

        invalid_mask = (tree_cover < 0) | (tree_cover > 100) | ~np.isfinite(tree_cover)
        tree_cover[invalid_mask] = 0.0
        n_invalid = int(invalid_mask.sum())

        n_with_cover = int((tree_cover > 0).sum())
        mean_cover = tree_cover[tree_cover > 0].mean() if n_with_cover > 0 else 0.0

        logger.info(f"Tree cover (direct pct): {n_with_cover} pixels with cover>0 "
                    f"({100*n_with_cover/canopy.size:.1f}% of area), "
                    f"mean={mean_cover:.1f}%, {n_invalid} invalid pixels -> 0%")

        if n_with_cover > 0:
            percentiles = np.percentile(tree_cover[tree_cover > 0], [10, 25, 50, 75, 90])
            logger.info(f"  Cover distribution p10/25/50/75/90: "
                        f"{percentiles[0]:.0f}/{percentiles[1]:.0f}/"
                        f"{percentiles[2]:.0f}/{percentiles[3]:.0f}/{percentiles[4]:.0f}%")
        else:
            logger.warning("  No pixels with tree cover > 0! "
                           "Check source type setting and canopy input.")

        return tree_cover

    elif source_type == "evc_encoded":
        tree_cover = np.zeros_like(canopy, dtype=np.float32)
        tree_mask = (canopy >= config.tree_cover_min_evc) & (canopy <= config.tree_cover_max_evc)
        tree_cover[tree_mask] = (canopy[tree_mask] - 100).astype(np.float32)

        n_tree = int(tree_mask.sum())
        logger.info(f"Tree cover (EVC encoded): {n_tree} pixels "
                    f"({100*n_tree/canopy.size:.1f}% of area)")
        return tree_cover

    else:
        raise ValueError(
            f"Unknown canopy_source_type: {source_type!r}. "
            f"Expected 'direct_pct' or 'evc_encoded'."
        )


def extract_shrub_cover_percent(canopy, config=None):
    # Extract shrub cover percentage (0-100) from canopy raster.
    # Meaningful only for LANDFIRE EVC (range 210-299); direct_pct
    # sources return zeros (shrub not separately encoded).
    # @param canopy: 2D numpy array of canopy raster values
    # @param config: VegetationConfig instance
    # @returns: numpy array - shrub cover percentage (0-100), float32

    if config is None:
        config = VegetationConfig()

    source_type = _get_source_type(config)

    if source_type == "direct_pct":
        shrub_cover = np.zeros_like(canopy, dtype=np.float32)
        logger.info("Shrub cover: N/A for direct_pct source (returning zeros)")
        return shrub_cover

    elif source_type == "evc_encoded":
        shrub_cover = np.zeros_like(canopy, dtype=np.float32)
        shrub_mask = (canopy >= config.shrub_cover_min_evc) & (canopy <= config.shrub_cover_max_evc)
        shrub_cover[shrub_mask] = (canopy[shrub_mask] - 200).astype(np.float32)

        n_shrub = int(shrub_mask.sum())
        logger.info(f"Shrub cover (EVC encoded): {n_shrub} pixels "
                    f"({100*n_shrub/canopy.size:.1f}% of area)")
        return shrub_cover

    else:
        raise ValueError(f"Unknown canopy_source_type: {source_type!r}")


# --------- FSI Computation ---------

def compute_fsi(tree_cover_pct, config=None):
    # Compute Forest Structure Index (0-1) from tree cover percentage,
    # used as input to the FlowPy forest friction / detrainment module.
    # @param tree_cover_pct: numpy array of tree cover % (0-100)
    # @param config: VegetationConfig instance
    # @returns: numpy array - FSI values (0 = no forest, 1 = full effect)

    if config is None:
        config = VegetationConfig()

    fsi = np.clip(tree_cover_pct / config.fsi_max_cover, 0.0, 1.0).astype(np.float32)

    n_forested = int((fsi > 0).sum())
    mean_fsi = float(fsi[fsi > 0].mean()) if n_forested > 0 else 0.0
    logger.info(f"FSI: {n_forested} forested pixels, mean FSI={mean_fsi:.2f} "
                f"(fsi_max_cover={config.fsi_max_cover}%)")
    return fsi


# --------- Reprojection ---------

def reproject_canopy_to_dem(canopy_data, dem_transform, dem_shape, dem_crs, config=None):
    # Reproject canopy raster to match DEM geometry. Resampling depends on
    # source type: direct_pct -> bilinear (continuous), evc_encoded ->
    # nearest (categorical codes must not be averaged).
    # @param canopy_data: dict from load_canopy()
    # @param dem_transform: Target affine transform
    # @param dem_shape: Target (height, width)
    # @param dem_crs: Target CRS
    # @param config: VegetationConfig instance
    # @returns: numpy array - reprojected canopy values (float32 for
    #           direct_pct, original dtype for evc_encoded)

    if config is None:
        config = VegetationConfig()

    source_type = _get_source_type(config)

    logger.info(f"Reprojecting canopy: source CRS={canopy_data['crs']} "
                f"-> DEM CRS={dem_crs}")
    if canopy_data['crs'] != dem_crs:
        logger.warning(f"  CRS MISMATCH - reprojecting on-the-fly. "
                       f"For best results, pre-warp canopy to DEM CRS.")

    if source_type == "direct_pct":
        canopy_reproj = np.empty(dem_shape, dtype=np.float32)
        reproject(
            source=canopy_data['data'],
            destination=canopy_reproj,
            src_transform=canopy_data['transform'],
            src_crs=canopy_data['crs'],
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
            src_nodata=canopy_data['nodata'],
            dst_nodata=-9999.0,
        )
        logger.info(f"Reprojected canopy (direct_pct, bilinear) to DEM grid ({dem_shape})")
        return canopy_reproj

    elif source_type == "evc_encoded":
        canopy_reproj = np.empty(dem_shape, dtype=canopy_data['data'].dtype)
        reproject(
            source=canopy_data['data'],
            destination=canopy_reproj,
            src_transform=canopy_data['transform'],
            src_crs=canopy_data['crs'],
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
            src_nodata=canopy_data['nodata'],
            dst_nodata=canopy_data['nodata'] if canopy_data['nodata'] is not None else -9999,
        )
        logger.info(f"Reprojected canopy (evc_encoded, nearest) to DEM grid ({dem_shape})")
        return canopy_reproj

    else:
        raise ValueError(f"Unknown canopy_source_type: {source_type!r}")


# --------- Main Workflow ---------

def create_fsi_for_dem(dem_path, canopy_path, config=None, output_path=None):
    # Create FSI raster matching DEM extent and resolution, plus tree
    # cover percentage for PRA detection. Supports LANDFIRE EVC and
    # direct-percent canopy sources.
    # @param dem_path: Path to DEM GeoTIFF
    # @param canopy_path: Path to canopy GeoTIFF (EVC or direct-percent)
    # @param config: VegetationConfig instance (controls source type)
    # @param output_path: Optional path to save FSI raster
    # @returns: dict with keys: fsi, tree_cover_pct, shrub_cover_pct,
    #           transform, crs, shape

    if config is None:
        config = VegetationConfig()

    source_type = _get_source_type(config)
    logger.info(f"Creating FSI from canopy source type: {source_type}")

    with rasterio.open(dem_path) as src:
        dem_transform = src.transform
        dem_shape = (src.height, src.width)
        dem_crs = src.crs
        pixel_size = src.res[0]

    canopy_data = load_canopy(canopy_path)
    canopy_reproj = reproject_canopy_to_dem(
        canopy_data, dem_transform, dem_shape, dem_crs, config
    )

    tree_cover_pct = extract_tree_cover_percent(canopy_reproj, config)
    shrub_cover_pct = extract_shrub_cover_percent(canopy_reproj, config)

    # Optional FSI smoothing (useful for blocky LANDFIRE input). PRA-specific
    # tree cover smoothing is separate and lives in release_detection.compute_pra.
    if config.veg_smooth_sigma_m > 0 and pixel_size > 0:
        from scipy.ndimage import gaussian_filter
        sigma_px = config.veg_smooth_sigma_m / pixel_size
        if sigma_px >= 0.5:
            tree_cover_pct = gaussian_filter(tree_cover_pct, sigma=sigma_px)
            shrub_cover_pct = gaussian_filter(shrub_cover_pct, sigma=sigma_px)
            tree_cover_pct = np.clip(tree_cover_pct, 0, 100)
            shrub_cover_pct = np.clip(shrub_cover_pct, 0, 100)
            logger.info(f"Smoothed vegetation cover (sigma={sigma_px:.1f}px, "
                        f"{config.veg_smooth_sigma_m:.0f}m)")

    fsi = compute_fsi(tree_cover_pct, config)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": dem_shape[0],
            "width": dem_shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": dem_crs,
            "transform": dem_transform,
            "nodata": -9999.0,
            "compress": "LZW"
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(fsi, 1)

        logger.info(f"Saved FSI raster: {output_path}")

    return {
        'fsi': fsi,
        'tree_cover_pct': tree_cover_pct,
        'shrub_cover_pct': shrub_cover_pct,
        'transform': dem_transform,
        'crs': dem_crs,
        'shape': dem_shape
    }
