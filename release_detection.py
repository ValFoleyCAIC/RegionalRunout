"""
Potential Release Area (PRA) Detection Module

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Identifies potential avalanche release areas with a fuzzy logic
    approach (Veitinger 2016) plus forest density weighting, producing a
    continuous PRA raster (0-1). A configurable threshold makes a binary
    mask, which is individualized by two-pass segmentation - pass 1
    watershed on the DEM (Duvillier et al. 2023), pass 2 aspect-weighted
    segmentation (Bühler et al. 2018) - then filtered by area and converted
    to polygons. Also handles graduated (sigmoid) vegetation weighting,
    optional tree-cover smoothing, optional wind shelter, sympathetic
    buffering, loading existing release areas from polygon or raster, and
    clipping label outputs to a core bbox for tiled runs.
"""

import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from scipy.ndimage import label, binary_erosion, binary_dilation
from shapely.geometry import shape as shapely_shape
import geopandas as gpd
from pathlib import Path
import logging
from datetime import datetime

from config import PRAConfig

logger = logging.getLogger(__name__)


# --------- Fuzzy Membership Functions ---------

def cauchy_membership(x, a, b, c):
    # Cauchy membership: mu(x) = 1 / (1 + ((x - c) / a)^(2b)).
    # @param x: Input values (numpy array or scalar)
    # @param a: Width parameter
    # @param b: Shape parameter (steepness)
    # @param c: Center parameter (peak location)
    # @returns: Membership values between 0 and 1

    return 1.0 / (1.0 + ((x - c) / a) ** (2 * b))


def forest_membership(tree_cover_pct, midpoint=40.0, steepness=10.0):
    # Forest density membership: higher cover -> lower release probability,
    # strictly monotonic (unlike a Cauchy centered at c=50, here 80% cover is
    # always more protective than 50%). Logistic sigmoid.
    # @param tree_cover_pct: Tree cover percentage (0-100)
    # @param midpoint: Cover % where membership = 0.5
    # @param steepness: Width of transition (higher = sharper dropoff)
    # @returns: Membership (0 = dense forest, 1 = open terrain)

    return 1.0 / (1.0 + np.exp((tree_cover_pct - midpoint) / steepness))


def fuzzy_and(membership_values, gamma=0.5):
    # Fuzzy AND operator (Werners 1988), combining membership layers.
    # @param membership_values: list of numpy arrays, each 0-1
    # @param gamma: 0=union, 0.5=balanced, 1=intersection
    # @returns: numpy array - combined membership

    product = np.ones_like(membership_values[0], dtype=np.float64)
    for mv in membership_values:
        product *= mv

    stack = np.stack(membership_values, axis=0)
    minimum = np.min(stack, axis=0)

    result = gamma * product + (1.0 - gamma) * minimum
    return result.astype(np.float32)


# --------- PRA Computation ---------

def compute_pra(slope_deg, tree_cover_pct, config, mask=None, wind_shelter=None,
                pixel_size=None):
    # Compute continuous PRA raster using fuzzy logic.
    # @param slope_deg: Slope in degrees
    # @param tree_cover_pct: Tree cover percentage (0-100)
    # @param config: PRAConfig instance
    # @param mask: Boolean nodata mask (True = nodata), optional
    # @param wind_shelter: Wind shelter index array, optional
    # @param pixel_size: Pixel size in meters - required when
    #        config.tree_cover_smooth_sigma_m > 0
    # @returns: numpy array - PRA values (0-1)

    logger.info("Computing fuzzy PRA...")

    # Optional tree-cover smoothing before fuzzy logic: at fine DEM
    # resolutions (1-3m) individual tree-shadow pixels punch through the
    # fuzzy AND ("swiss cheese" polygons); smoothing averages trees into
    # local stand density.
    sigma_m = getattr(config, "tree_cover_smooth_sigma_m", 0.0)
    if sigma_m > 0 and pixel_size is not None and pixel_size > 0:
        sigma_px = sigma_m / pixel_size
        if sigma_px >= 0.5:
            from scipy.ndimage import gaussian_filter
            tree_cover_smoothed = gaussian_filter(
                tree_cover_pct.astype(np.float32), sigma=sigma_px
            )
            tree_cover_smoothed = np.clip(tree_cover_smoothed, 0.0, 100.0)
            logger.info(f"  Smoothed tree cover (sigma={sigma_px:.1f}px, "
                        f"{sigma_m:.0f}m) before fuzzy logic")
            tree_cover_pct = tree_cover_smoothed

    mu_slope = cauchy_membership(slope_deg, config.slope_a, config.slope_b, config.slope_c)
    logger.info(f"  Slope membership: mean={mu_slope[mu_slope > 0].mean():.3f}")

    mu_forest = forest_membership(
        tree_cover_pct,
        midpoint=config.forest_midpoint,
        steepness=config.forest_steepness
    )
    logger.info(f"  Forest membership: mean={mu_forest[mu_forest > 0].mean():.3f}")

    memberships = [mu_slope, mu_forest]

    if config.use_wind_shelter and wind_shelter is not None:
        mu_wind = cauchy_membership(
            wind_shelter, config.wind_shelter_a, config.wind_shelter_b, config.wind_shelter_c
        )
        memberships.append(mu_wind)
        logger.info(f"  Wind shelter membership: mean={mu_wind[mu_wind > 0].mean():.3f}")

    pra = fuzzy_and(memberships, gamma=config.fuzzy_gamma)

    if mask is not None:
        pra[mask] = 0.0

    logger.info(f"  PRA range: [{pra.min():.4f}, {pra.max():.4f}]")
    logger.info(f"  Pixels above threshold ({config.pra_threshold}): "
                f"{(pra >= config.pra_threshold).sum()}")

    return pra


# --------- Binary Conversion ---------

def pra_to_binary(pra, config, dem=None, mask=None, slope_deg=None, pixel_size=None):
    # Convert continuous PRA to a binary release mask: threshold, elevation
    # filter, morphological cleanup, sympathetic buffer.
    # @param pra: Continuous PRA array (0-1)
    # @param config: PRAConfig instance
    # @param dem: Elevation array (optional, for elevation filtering)
    # @param mask: Boolean nodata mask (optional)
    # @param slope_deg: Slope array (optional, for terrain-aware buffer)
    # @param pixel_size: Pixel size in meters (needed for buffer/patch filtering)
    # @returns: numpy array - boolean release mask (True = release area)

    logger.info("Converting PRA to binary release areas...")

    release = pra >= config.pra_threshold
    logger.info(f"  After threshold ({config.pra_threshold}): {release.sum()} pixels")

    if dem is not None:
        if config.min_elevation_m is not None:
            release &= dem >= config.min_elevation_m
        if config.max_elevation_m is not None:
            release &= dem <= config.max_elevation_m
        logger.info(f"  After elevation filter: {release.sum()} pixels")

    if mask is not None:
        release &= ~mask

    if config.erosion_iterations > 0:
        release = binary_erosion(release, iterations=config.erosion_iterations)
    if config.dilation_iterations > 0:
        release = binary_dilation(release, iterations=config.dilation_iterations)

    # Sympathetic buffer: fixed distance
    if config.sympathetic_buffer_m > 0 and pixel_size is not None:
        buffer_pixels = max(1, int(config.sympathetic_buffer_m / pixel_size))
        release = binary_dilation(release, iterations=buffer_pixels)
        logger.info(f"  After sympathetic buffer ({config.sympathetic_buffer_m}m): {release.sum()} pixels")

    # Sympathetic buffer: terrain-aware expansion
    if config.terrain_aware_buffer and slope_deg is not None:
        release = terrain_aware_expand(
            release, slope_deg, config.terrain_buffer_min_slope_deg,
            config.terrain_buffer_max_iterations
        )
        logger.info(f"  After terrain-aware buffer: {release.sum()} pixels")

    # Re-apply mask after buffering
    if mask is not None:
        release &= ~mask

    logger.info(f"  Final release pixels: {release.sum()}")
    return release


def terrain_aware_expand(release_mask, slope_deg, min_slope_deg, max_iterations):
    # Expand release areas into adjacent cells meeting a relaxed slope threshold.
    # @param release_mask: Current binary release mask
    # @param slope_deg: Slope in degrees
    # @param min_slope_deg: Minimum slope for expansion
    # @param max_iterations: Maximum expansion iterations
    # @returns: Expanded release mask

    slope_eligible = slope_deg >= min_slope_deg
    expanded = release_mask.copy()

    for i in range(max_iterations):
        candidate = binary_dilation(expanded, iterations=1)
        # Keep only new pixels that meet slope criteria
        new_pixels = candidate & ~expanded & slope_eligible
        if new_pixels.sum() == 0:
            break
        expanded |= new_pixels

    return expanded


# --------- Watershed + Aspect PRA Individualization ---------
# Two-pass: pass 1 watershed on the DEM (Duvillier et al. 2023) splits terrain
# along ridgelines; pass 2 aspect-weighted segmentation (Bühler et al. 2018)
# subdivides large basins along aspect discontinuities. Intersect with the
# binary PRA mask, then area-filter.

def compute_watershed_basins(dem, mask, pixel_size, basin_target_size_m=200.0):
    # Delineate watershed basins from a DEM. Inverts the DEM (ridgelines
    # become valleys), then floods from ridge markers with scikit-image's
    # watershed, assigning each cell to its nearest ridgeline (≈ D8).
    # @param dem: 2D elevation array (smoothed)
    # @param mask: Boolean nodata mask (True = nodata)
    # @param pixel_size: Pixel size in meters
    # @param basin_target_size_m: Target basin dimension; sets the local-max
    #        filter size for ridge markers. Smaller = more basins.
    # @returns: 2D int array of basin labels (0 = nodata)

    from skimage.segmentation import watershed
    from scipy.ndimage import maximum_filter, label as ndlabel

    logger.info("  Computing watershed basins...")

    # Inverted DEM surface
    inverted = np.empty_like(dem, dtype=np.float32)
    inverted[~mask] = -dem[~mask]
    inverted[mask] = np.finfo(np.float32).max

    # Ridge markers
    filter_pixels = max(3, int(basin_target_size_m / pixel_size))
    if filter_pixels % 2 == 0:
        filter_pixels += 1

    local_max = maximum_filter(dem, size=filter_pixels)
    ridge_markers = (dem == local_max) & ~mask
    ridge_labeled, n_ridges = ndlabel(ridge_markers)

    if n_ridges == 0:
        logger.warning("  No ridge markers found — returning single basin")
        basins = np.zeros_like(dem, dtype=np.int32)
        basins[~mask] = 1
        return basins

    logger.info(f"  Found {n_ridges} ridge markers (filter={filter_pixels}px, "
                f"~{filter_pixels * pixel_size:.0f}m)")

    basins = watershed(inverted, markers=ridge_labeled, mask=~mask)

    n_basins = len(np.unique(basins[basins > 0]))
    logger.info(f"  Delineated {n_basins} watershed basins")

    return basins.astype(np.int32)


def _classify_aspect_sector(aspect_deg):
    # Classify aspect into 8 cardinal sectors (45-degree bins).
    # Sectors: 0=N(337.5-22.5), 1=NE(22.5-67.5), ... 7=NW(292.5-337.5).
    # Nodata (-9999) gets sector -1.
    # @param aspect_deg: 2D aspect in degrees (0=N, 90=E, 180=S, 270=W)
    # @returns: 2D int array of sector labels (0-7, or -1 for nodata)

    sectors = np.full_like(aspect_deg, -1, dtype=np.int8)
    valid = aspect_deg >= 0

    # Shift so N is centered: (aspect + 22.5) mod 360 / 45
    shifted = (aspect_deg[valid] + 22.5) % 360.0
    sectors[valid] = (shifted / 45.0).astype(np.int8)

    return sectors


def segment_by_aspect(basins, aspect_deg, slope_deg, mask, pixel_size,
                      aspect_weight=3.0, segment_target_size_m=75.0):
    # Second-pass segmentation splitting basins along aspect discontinuities
    # (Bühler et al. 2018): build a composite of aspect discontinuity
    # (weighted aspect_weight x) plus slope variation, then watershed it to
    # subdivide basins whose slopes face different directions - e.g. Colorado
    # bowls with NE and NW faces that should be separate release areas.
    # @param basins: 2D basin labels from compute_watershed_basins
    # @param aspect_deg: 2D aspect (0=N, 90=E, degrees)
    # @param slope_deg: 2D slope (degrees)
    # @param mask: Boolean nodata mask
    # @param pixel_size: Pixel size in meters
    # @param aspect_weight: Aspect vs slope weight (Bühler et al. 2018 used 3.0)
    # @param segment_target_size_m: Target aspect-segmentation size; smaller =
    #        more splits. 75m catches minor ribs in open bowls.
    # @returns: 2D int array of refined basin labels

    from skimage.segmentation import watershed
    from scipy.ndimage import label as ndlabel
    from scipy.ndimage import uniform_filter

    logger.info("  Refining basins by aspect segmentation (Bühler et al. 2018)...")

    # Aspect discontinuity surface. Use sin/cos to handle the 360/0 wrap;
    # the gradient magnitude of those components is the aspect change rate.
    aspect_rad = np.radians(np.where(aspect_deg >= 0, aspect_deg, 0.0))
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)

    dsin_y, dsin_x = np.gradient(sin_aspect, pixel_size, pixel_size)
    dcos_y, dcos_x = np.gradient(cos_aspect, pixel_size, pixel_size)
    aspect_gradient = np.sqrt(dsin_x**2 + dsin_y**2 + dcos_x**2 + dcos_y**2)

    # Slope gradient = slope change rate (curvature-like)
    dslope_y, dslope_x = np.gradient(slope_deg, pixel_size, pixel_size)
    slope_gradient = np.sqrt(dslope_x**2 + dslope_y**2)

    # Normalize both to [0, 1]
    ag_valid = aspect_gradient[~mask]
    sg_valid = slope_gradient[~mask]

    if len(ag_valid) > 0 and ag_valid.max() > 0:
        aspect_norm = aspect_gradient / np.percentile(ag_valid, 99)
    else:
        aspect_norm = aspect_gradient

    if len(sg_valid) > 0 and sg_valid.max() > 0:
        slope_norm = slope_gradient / np.percentile(sg_valid, 99)
    else:
        slope_norm = slope_gradient

    aspect_norm = np.clip(aspect_norm, 0, 1)
    slope_norm = np.clip(slope_norm, 0, 1)

    # Composite: aspect weighted aspect_weight x more than slope
    composite = (aspect_weight * aspect_norm + slope_norm) / (aspect_weight + 1.0)
    composite[mask] = 0.0

    # Slight smoothing to avoid over-segmentation from noise
    smooth_px = max(1, int(15.0 / pixel_size))
    if smooth_px >= 2:
        composite = uniform_filter(composite, size=smooth_px)
        composite[mask] = 0.0

    # Seeds = local minima of the composite (smooth, homogeneous terrain);
    # high-discontinuity ridges become watershed boundaries.
    seg_filter_px = max(3, int(segment_target_size_m / pixel_size))
    if seg_filter_px % 2 == 0:
        seg_filter_px += 1

    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(composite, size=seg_filter_px)
    homogeneous_seeds = (composite == local_min) & ~mask & (composite < 0.5)

    seed_labeled, n_seeds = ndlabel(homogeneous_seeds)

    if n_seeds <= 1:
        logger.info("  No aspect discontinuities found — keeping watershed basins")
        return basins

    logger.info(f"  Found {n_seeds} aspect-homogeneous seeds "
                f"(filter={seg_filter_px}px, ~{seg_filter_px * pixel_size:.0f}m)")

    composite_int = (composite * 65534).astype(np.float32)
    composite_int[mask] = np.finfo(np.float32).max

    aspect_basins = watershed(composite_int, markers=seed_labeled, mask=~mask)

    # Combine (original_basin, aspect_basin) pairs
    max_aspect = int(aspect_basins.max()) + 1
    combined = np.where(
        ~mask,
        basins.astype(np.int64) * max_aspect + aspect_basins.astype(np.int64),
        0
    )

    # Relabel to sequential
    flat = combined.ravel()
    unique_vals, inverse = np.unique(flat, return_inverse=True)
    remap = np.arange(len(unique_vals), dtype=np.int32)
    if len(unique_vals) > 0 and unique_vals[0] == 0:
        remap[0] = 0

    refined = remap[inverse].reshape(combined.shape)

    n_refined = int(refined.max())
    n_original = len(np.unique(basins[basins > 0]))
    logger.info(f"  Aspect segmentation: {n_original} basins -> {n_refined} refined basins")

    return refined


def individualize_pra(release_mask, dem, slope_deg, aspect_deg, mask,
                      pixel_size, basin_target_size_m=200.0,
                      aspect_weight=3.0, segment_target_size_m=75.0):
    # Split a binary PRA mask into individual release areas via two-pass
    # segmentation (watershed then aspect).
    # @param release_mask: Boolean PRA mask (True = release area)
    # @param dem: 2D elevation array (smoothed)
    # @param slope_deg: 2D slope array (degrees)
    # @param aspect_deg: 2D aspect array (degrees, 0=N)
    # @param mask: Boolean nodata mask
    # @param pixel_size: Pixel size in meters
    # @param basin_target_size_m: Target size for watershed pass (meters)
    # @param aspect_weight: Aspect vs slope weight in pass 2 (Bühler: 3.0)
    # @param segment_target_size_m: Target size for aspect pass (meters)
    # @returns: 2D int array - each unique positive value is one release area

    logger.info("Individualizing PRAs via watershed + aspect segmentation...")

    basins = compute_watershed_basins(dem, mask, pixel_size, basin_target_size_m)

    refined = segment_by_aspect(
        basins, aspect_deg, slope_deg, mask, pixel_size,
        aspect_weight=aspect_weight,
        segment_target_size_m=segment_target_size_m
    )

    # Intersect with PRA mask
    connected, n_connected = label(release_mask)

    max_refined = int(refined.max()) + 1
    composite = np.where(
        release_mask,
        connected.astype(np.int64) * max_refined + refined.astype(np.int64),
        0
    )

    # Vectorized relabeling
    flat = composite.ravel()
    unique_vals, inverse = np.unique(flat, return_inverse=True)

    remap = np.arange(len(unique_vals), dtype=np.int32)
    if len(unique_vals) > 0 and unique_vals[0] == 0:
        remap[0] = 0
        n_pra = len(unique_vals) - 1
    else:
        n_pra = len(unique_vals)

    individualized = remap[inverse].reshape(composite.shape)

    logger.info(f"  {n_connected} connected PRA regions -> "
                f"{n_pra} individualized PRAs")

    return individualized


def filter_patches(release_mask, pixel_size, config, dem=None, mask=None,
                   slope_deg=None, aspect_deg=None,
                   basin_target_size_m=200.0, aspect_weight=3.0,
                   segment_target_size_m=75.0):
    # Individualize and area-filter release patches. With DEM+slope+aspect,
    # uses full two-pass segmentation; with DEM only, watershed only; with
    # nothing, connected-component labeling.
    # @param release_mask: Boolean release area mask
    # @param pixel_size: Pixel size in meters
    # @param config: PRAConfig instance
    # @param dem: 2D elevation array (optional but recommended)
    # @param mask: Boolean nodata mask
    # @param slope_deg: 2D slope array (optional, needed for aspect pass)
    # @param aspect_deg: 2D aspect array (optional, needed for aspect pass)
    # @param basin_target_size_m: Target basin size for watershed (meters)
    # @param aspect_weight: Aspect vs slope weight for pass 2 (Bühler: 3.0)
    # @param segment_target_size_m: Target size for aspect segmentation (meters)
    # @returns: Filtered int32 label array (dropped patches = 0)

    pixel_area_m2 = pixel_size * pixel_size

    # --- Individualize ---
    if dem is not None and mask is not None and slope_deg is not None and aspect_deg is not None:
        individualized = individualize_pra(
            release_mask, dem, slope_deg, aspect_deg, mask, pixel_size,
            basin_target_size_m=basin_target_size_m,
            aspect_weight=aspect_weight,
            segment_target_size_m=segment_target_size_m
        )
    elif dem is not None and mask is not None:
        logger.info("  No aspect data — using watershed only (no aspect refinement)")
        basins = compute_watershed_basins(dem, mask, pixel_size, basin_target_size_m)
        connected, n_connected = label(release_mask)
        max_b = int(basins.max()) + 1
        comp = np.where(release_mask,
                        connected.astype(np.int64) * max_b + basins.astype(np.int64), 0)
        flat = comp.ravel()
        uv, inv = np.unique(flat, return_inverse=True)
        remap = np.arange(len(uv), dtype=np.int32)
        if len(uv) > 0 and uv[0] == 0:
            remap[0] = 0
        individualized = remap[inv].reshape(comp.shape)
    else:
        logger.info("  No DEM provided — using connected-component labeling only")
        individualized, _ = label(release_mask)

    n_total = individualized.max()
    if n_total == 0:
        logger.warning("  No patches found")
        return np.zeros_like(release_mask, dtype=bool)

    # --- Area filtering (vectorized) ---
    logger.info(f"  Filtering {n_total} individualized patches by area...")

    patch_ids = np.arange(1, n_total + 1)
    patch_pixel_counts = np.bincount(individualized.ravel(), minlength=n_total + 1)
    patch_areas = patch_pixel_counts * pixel_area_m2

    keep_ids = patch_ids[patch_areas[patch_ids] >= config.min_patch_area_m2]
    n_oversized = int(np.sum(patch_areas[patch_ids] > config.max_patch_area_m2))

    keep_mask = np.isin(individualized, keep_ids)
    # Keep the labeled array (dropped patches -> 0) so each release zone
    # stays a distinct feature in the polygon export.
    filtered_labels = np.where(keep_mask, individualized, 0).astype(np.int32)

    dropped_small = n_total - len(keep_ids)

    logger.info(f"  Kept {len(keep_ids)}/{n_total} patches "
                f"(min={config.min_patch_area_m2:.0f} m2)")
    if dropped_small > 0:
        logger.info(f"  Dropped {dropped_small} patches below min area")
    if n_oversized > 0:
        logger.info(f"  {n_oversized} patches exceed {config.max_patch_area_m2:.0f} m2 "
                     f"(kept — single-basin PRAs)")

    return filtered_labels


# --------- Existing Release Area Loading ---------

def load_existing_release_areas(release_path, dem_path):
    # Load existing release areas from polygon or raster. Polygons are
    # rasterized to the DEM grid; rasters are read directly.
    # @param release_path: Path to release area file (.shp/.gpkg/.tif)
    # @param dem_path: Path to DEM (for grid alignment)
    # @returns: numpy array - boolean release mask (True = release area)

    release_path = Path(release_path)

    with rasterio.open(dem_path) as src:
        dem_shape = (src.height, src.width)
        dem_transform = src.transform
        dem_crs = src.crs

    ext = release_path.suffix.lower()

    if ext in ['.tif', '.tiff']:
        logger.info(f"Loading release raster: {release_path.name}")
        with rasterio.open(release_path) as src:
            data = src.read(1)
        release_mask = data > 0
        logger.info(f"  Loaded {release_mask.sum()} release pixels from raster")

    elif ext in ['.shp', '.gpkg', '.geojson']:
        logger.info(f"Loading release polygons: {release_path.name}")
        gdf = gpd.read_file(release_path)

        if gdf.crs != dem_crs:
            logger.info(f"  Reprojecting from {gdf.crs} to {dem_crs}")
            gdf = gdf.to_crs(dem_crs)

        logger.info(f"  {len(gdf)} features loaded")

        shapes_list = [(geom, 1) for geom in gdf.geometry if geom is not None]
        release_data = rasterize(
            shapes=shapes_list,
            out_shape=dem_shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8
        )
        release_mask = release_data > 0
        logger.info(f"  Rasterized to {release_mask.sum()} release pixels")

    else:
        raise ValueError(f"Unsupported release area format: {ext}")

    return release_mask


# --------- Vector Conversion ---------

def clip_labels_to_bbox(release_labels, transform, bbox):
    # Zero out labeled pixels outside a world-coordinate bbox. Used by the
    # tile_processor to drop the overlap region so adjacent tiles don't
    # double up at seams.
    # @param release_labels: 2D int array (0 = background, >0 = patch labels)
    # @param transform: rasterio affine transform of the array
    # @param bbox: (minx, miny, maxx, maxy) in the array's CRS
    # @returns: 2D int array with labels outside bbox set to 0

    from rasterio.windows import from_bounds

    arr = np.asarray(release_labels)
    h, w = arr.shape

    minx, miny, maxx, maxy = bbox

    win = from_bounds(minx, miny, maxx, maxy, transform=transform)
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))
    col_end = int(np.ceil(win.col_off + win.width))
    row_end = int(np.ceil(win.row_off + win.height))

    # Clamp to array bounds
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(w, col_end)
    row_end = min(h, row_end)

    keep = np.zeros_like(arr, dtype=bool)
    if col_end > col_off and row_end > row_off:
        keep[row_off:row_end, col_off:col_end] = True

    n_before = int((arr > 0).sum())
    clipped = np.where(keep, arr, 0).astype(arr.dtype)
    n_after = int((clipped > 0).sum())
    pct = (100.0 * n_after / n_before) if n_before else 0.0
    logger.info(f"  Core-bbox clip: kept {n_after:,}/{n_before:,} release pixels ({pct:.1f}%)")

    return clipped


def mask_to_polygons(release, transform, crs):
    # Convert a release array to a polygon GeoDataFrame. Integer label arrays
    # give one polygon per non-zero label (preserving segmentation); boolean
    # arrays merge connected True pixels into one polygon.
    # @param release: Boolean mask OR integer label array (0 = background)
    # @param transform: Rasterio affine transform
    # @param crs: CRS
    # @returns: GeoDataFrame with one polygon per release zone

    arr = np.asarray(release)
    if arr.dtype == bool:
        as_int = arr.astype(np.int32)
    else:
        as_int = arr.astype(np.int32, copy=False)

    polygons = []
    areas = []
    labels = []

    for geom_dict, value in shapes(as_int, transform=transform):
        if value == 0:
            continue
        geom = shapely_shape(geom_dict)
        polygons.append(geom)
        areas.append(geom.area)
        labels.append(int(value))

    if not polygons:
        logger.warning("No polygons generated from release array")
        return gpd.GeoDataFrame(columns=['release_id', 'geometry', 'area_m2'], crs=crs)

    gdf = gpd.GeoDataFrame(
        {'release_id': labels, 'geometry': polygons, 'area_m2': areas},
        crs=crs
    )

    logger.info(f"Generated {len(gdf)} release area polygons "
                f"({len(set(labels))} unique IDs)")
    return gdf


# --------- Save Outputs ---------

def save_release_outputs(release_mask, pra, transform, crs, output_dir, file_stem):
    # Save release area raster, vector, and continuous PRA.
    # @param release_mask: Boolean or label release mask
    # @param pra: Continuous PRA array (0-1), or None
    # @param transform: Rasterio affine transform
    # @param crs: CRS
    # @param output_dir: Output directory
    # @param file_stem: Filename stem
    # @returns: dict with output file paths

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Binary release raster (any non-zero label = release)
    raster_path = output_dir / f"{file_stem}_release_areas.tif"
    profile = {
        "driver": "GTiff",
        "height": release_mask.shape[0],
        "width": release_mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "LZW"
    }
    binary_mask = (release_mask > 0).astype(np.uint8)
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(binary_mask, 1)
    outputs['release_raster'] = str(raster_path)
    logger.info(f"Saved release raster: {raster_path.name}")

    # Continuous PRA raster
    if pra is not None:
        pra_path = output_dir / f"{file_stem}_pra_continuous.tif"
        pra_profile = profile.copy()
        pra_profile.update({"dtype": "float32", "nodata": -9999.0})
        with rasterio.open(pra_path, "w", **pra_profile) as dst:
            dst.write(pra.astype(np.float32), 1)
        outputs['pra_raster'] = str(pra_path)
        logger.info(f"Saved continuous PRA: {pra_path.name}")

    # Release polygons
    gdf = mask_to_polygons(release_mask, transform, crs)
    if len(gdf) > 0:
        gpkg_path = output_dir / f"{file_stem}_release_areas.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG", layer="release_areas")
        outputs['release_vector'] = str(gpkg_path)

        shp_path = output_dir / f"{file_stem}_release_areas.shp"
        gdf.to_file(shp_path)
        outputs['release_shapefile'] = str(shp_path)

        logger.info(f"Saved {len(gdf)} release polygons")

    return outputs


# --------- Statistics ---------

def compute_release_statistics(release_mask, slope_deg, dem, transform, pixel_size):
    # Compute summary statistics for detected release areas.
    # @param release_mask: Boolean release mask
    # @param slope_deg: Slope array
    # @param dem: Elevation array
    # @param transform: Affine transform
    # @param pixel_size: Pixel size in meters
    # @returns: dict with statistics

    pixel_area_m2 = pixel_size * pixel_size
    release_pixels = release_mask.sum()
    release_area_m2 = release_pixels * pixel_area_m2

    if release_pixels == 0:
        # Return a complete dict so downstream can unconditionally read
        # stats['release_area_km2'] etc.
        return {
            "timestamp": datetime.now().isoformat(),
            "release_pixels": 0,
            "release_area_m2": 0.0,
            "release_area_km2": 0.0,
            "slope": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "elevation": {"mean": 0.0, "min": 0.0, "max": 0.0},
            "patches": {"count": 0, "areas_m2": [], "min_area_m2": 0.0,
                        "max_area_m2": 0.0, "mean_area_m2": 0.0},
        }

    release_slopes = slope_deg[release_mask]
    release_elevations = dem[release_mask]

    labeled, n_patches = label(release_mask)
    patch_areas = []
    for pid in range(1, n_patches + 1):
        patch_areas.append((labeled == pid).sum() * pixel_area_m2)

    stats = {
        "timestamp": datetime.now().isoformat(),
        "release_pixels": int(release_pixels),
        "release_area_m2": float(release_area_m2),
        "release_area_km2": float(release_area_m2 / 1e6),
        "slope": {
            "mean": float(release_slopes.mean()),
            "median": float(np.median(release_slopes)),
            "min": float(release_slopes.min()),
            "max": float(release_slopes.max()),
        },
        "elevation": {
            "mean": float(release_elevations.mean()),
            "min": float(release_elevations.min()),
            "max": float(release_elevations.max()),
        },
        "patches": {
            "count": n_patches,
            "mean_area_m2": float(np.mean(patch_areas)) if patch_areas else 0.0,
            "max_area_m2": float(np.max(patch_areas)) if patch_areas else 0.0,
        }
    }

    return stats
