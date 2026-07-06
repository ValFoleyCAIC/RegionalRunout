"""
Terrain Analysis Module

Author: Valerie Foley
Last Updated: 2/2026

Description:
    DEM loading, smoothing, slope, aspect, curvature, and wind shelter
    calculations for avalanche terrain analysis.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling as RioResampling
from scipy.ndimage import gaussian_filter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# --------- DEM Resampling ---------

def resample_dem(input_path, output_path, target_resolution_m):
    # Resample a DEM to a coarser (or finer) resolution using bilinear
    # interpolation to preserve smooth terrain surfaces.
    # @param input_path: Path to source DEM GeoTIFF
    # @param output_path: Path to write resampled DEM
    # @param target_resolution_m: Desired pixel size in meters
    # @returns: Path to resampled DEM

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        native_res = src.res[0]

        if abs(native_res - target_resolution_m) < 0.01:
            logger.info(f"DEM already at {native_res:.2f}m — skipping resample")
            return input_path

        scale_factor = native_res / target_resolution_m
        new_height = max(1, int(src.height * scale_factor))
        new_width = max(1, int(src.width * scale_factor))

        new_transform = rasterio.transform.from_bounds(
            *src.bounds, new_width, new_height
        )

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=RioResampling.bilinear,
        )

        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "compress": "LZW",
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    logger.info(f"Resampled DEM: {native_res:.2f}m -> {target_resolution_m:.2f}m "
                f"({src.height}x{src.width} -> {new_height}x{new_width})")

    return output_path


# --------- DEM Loading ---------

def load_dem(dem_path, smooth_sigma=1.5):
    # Load DEM and return smoothed data with metadata.
    # @param dem_path: Path to DEM GeoTIFF
    # @param smooth_sigma: Gaussian smoothing sigma (pixels), 0 = no smoothing
    # @returns: dict with keys: dem, dem_raw, transform, crs, nodata, pixel_size,
    #           bounds, shape, mask

    dem_path = Path(dem_path)

    with rasterio.open(dem_path) as src:
        dem_raw = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        pixel_size = src.res[0]
        bounds = src.bounds

    if nodata is not None:
        mask = (dem_raw == nodata) | ~np.isfinite(dem_raw)
    else:
        mask = ~np.isfinite(dem_raw)

    # Smooth DEM, preserving nodata cells
    if smooth_sigma > 0:
        dem = gaussian_filter(dem_raw, sigma=smooth_sigma)
        dem[mask] = dem_raw[mask]
    else:
        dem = dem_raw.copy()

    logger.info(f"Loaded DEM: {dem_path.name}")
    logger.info(f"  Shape: {dem_raw.shape}, Resolution: {pixel_size:.2f}m")

    return {
        'dem': dem,
        'dem_raw': dem_raw,
        'mask': mask,
        'transform': transform,
        'crs': crs,
        'nodata': nodata,
        'pixel_size': pixel_size,
        'bounds': bounds,
        'shape': dem_raw.shape
    }


# --------- Slope and Aspect ---------

def compute_slope_aspect(dem, pixel_size, mask):
    # Compute slope (degrees) and aspect (degrees from north, 0-360).
    # @param dem: 2D numpy array of smoothed elevation values
    # @param pixel_size: Pixel size in meters
    # @param mask: Boolean nodata mask (True = nodata)
    # @returns: tuple (slope_deg, aspect_deg)

    dz_dy, dz_dx = np.gradient(dem, pixel_size, pixel_size)

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad)

    # Aspect: 0=N, 90=E, 180=S, 270=W
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360.0

    slope_deg[mask] = 0.0
    aspect_deg[mask] = -9999.0

    logger.info("Computed slope and aspect")
    return slope_deg, aspect_deg


# --------- Curvature ---------

def compute_plan_curvature(dem, pixel_size, mask):
    # Compute plan (contour) curvature.
    # Positive = concave (collecting), negative = convex (dispersing).
    # @param dem: 2D numpy array of smoothed elevation values
    # @param pixel_size: Pixel size in meters
    # @param mask: Boolean nodata mask
    # @returns: numpy array - plan curvature (1/m)

    dz_dy, dz_dx = np.gradient(dem, pixel_size, pixel_size)
    dz_dyy = np.gradient(dz_dy, pixel_size, axis=0)
    dz_dxx = np.gradient(dz_dx, pixel_size, axis=1)
    dz_dxy = np.gradient(dz_dx, pixel_size, axis=0)

    p = dz_dx**2 + dz_dy**2
    q = np.sqrt(p + 1e-10)
    plan_curv = (dz_dxx * dz_dy**2 - 2*dz_dxy*dz_dx*dz_dy + dz_dyy*dz_dx**2) / (p*q + 1e-10)
    plan_curv[mask] = 0.0

    logger.info("Computed plan curvature")
    return plan_curv


# --------- Wind Shelter ---------

def compute_wind_shelter(dem, pixel_size, mask, wind_direction_deg=270.0, search_distance_m=300.0):
    # Compute wind shelter index (Winstral et al. 2002 Sx parameter).
    # Higher values = more sheltered (lee side, snow loading zone).
    # @param dem: 2D numpy array of smoothed elevation values
    # @param pixel_size: Pixel size in meters
    # @param mask: Boolean nodata mask
    # @param wind_direction_deg: Wind comes FROM this direction (deg from N)
    # @param search_distance_m: Max search distance upwind (meters)
    # @returns: numpy array - wind shelter index (positive = sheltered)

    nrows, ncols = dem.shape
    search_pixels = int(search_distance_m / pixel_size)

    wind_rad = np.radians(wind_direction_deg)
    dr = -np.cos(wind_rad)
    dc = np.sin(wind_rad)

    shelter = np.zeros_like(dem)

    for i in range(nrows):
        for j in range(ncols):
            if mask[i, j]:
                continue

            max_angle = -np.inf
            z_here = dem[i, j]

            for step in range(1, search_pixels + 1):
                ri = int(round(i + step * dr))
                ci = int(round(j + step * dc))

                if ri < 0 or ri >= nrows or ci < 0 or ci >= ncols:
                    break
                if mask[ri, ci]:
                    continue

                dist = step * pixel_size
                angle = np.degrees(np.arctan2(dem[ri, ci] - z_here, dist))
                if angle > max_angle:
                    max_angle = angle

            shelter[i, j] = max_angle if max_angle > -np.inf else 0.0

    shelter[mask] = 0.0
    logger.info(f"Computed wind shelter (direction: {wind_direction_deg} deg)")
    return shelter


# --------- Convenience Function ---------

def compute_all_terrain(dem_path, smooth_sigma=1.5, compute_curvature=True,
                        wind_shelter_config=None):
    # Compute all terrain derivatives from a DEM.
    # @param dem_path: Path to DEM GeoTIFF
    # @param smooth_sigma: DEM smoothing sigma
    # @param compute_curvature: Whether to compute plan curvature
    # @param wind_shelter_config: dict with wind_direction_deg and
    #        search_distance_m, or None
    # @returns: dict with dem data and all computed derivatives

    dem_data = load_dem(dem_path, smooth_sigma)

    slope_deg, aspect_deg = compute_slope_aspect(
        dem_data['dem'], dem_data['pixel_size'], dem_data['mask']
    )

    result = {**dem_data, 'slope_deg': slope_deg, 'aspect_deg': aspect_deg}

    if compute_curvature:
        plan_curv = compute_plan_curvature(
            dem_data['dem'], dem_data['pixel_size'], dem_data['mask']
        )
        result['plan_curvature'] = plan_curv

    if wind_shelter_config is not None:
        wind_shelter = compute_wind_shelter(
            dem_data['dem'],
            dem_data['pixel_size'],
            dem_data['mask'],
            wind_direction_deg=wind_shelter_config.get('wind_direction_deg', 270.0),
            search_distance_m=wind_shelter_config.get('search_distance_m', 300.0)
        )
        result['wind_shelter'] = wind_shelter

    return result


# --------- Raster Saving ---------

def save_raster(output_path, array, transform, crs, nodata=-9999.0, dtype="float32"):
    # Save numpy array as GeoTIFF.
    # @param output_path: Output file path
    # @param array: 2D numpy array
    # @param transform: Rasterio affine transform
    # @param crs: Coordinate reference system
    # @param nodata: NoData value
    # @param dtype: Output data type
    # @returns: None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "LZW",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

    logger.info(f"Saved raster: {output_path.name}")
