"""
Avalanche Path Extraction Module

Author: Valerie Foley
Last Updated: 7/2026

Description:
    Converts FlowPy simulation outputs (flux, cell counts, zdelta) into
    individualized avalanche path polygons with attributes. Follows the
    polygon-export idea of Bühler et al. (2022, NHESS), thresholding
    simulation output at a physically meaningful cutoff; for FlowPy the flux
    output is the equivalent, representing routed mass concentration that
    channels into gullies and fans at deposition (D'Amboise et al. 2022, GMD).
    Paths are made by thresholding flux, optionally tagging each flow region
    with its source release area, polygonizing connected regions, and
    computing attributes (area, elevation drop, flux stats). A single-polygon
    runout boundary (the union of all path pixels above threshold) is emitted
    as a hazard-extent footprint.
"""

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# --------- Configuration ---------

@dataclass
class PathExtractionConfig:
    # Configuration for path polygon extraction
    # Cells with flux >= this are part of a path. FlowPy's internal
    # flux_threshold (0.003) defines the runout limit; higher values here
    # give tighter paths:
    #   0.003 = full extent (matches FlowPy runout limit)
    #   0.01  = moderate trimming of marginal areas
    #   0.03  = core paths only (good for visualization)
    flux_threshold: float = 0.003

    # Minimum path area (m2) — remove tiny fragments
    min_path_area_m2: float = 500.0

    # Merge path fragments that share the same release area
    merge_by_release: bool = True

    # Polygon simplify tolerance (m). 0 = none; 1-5m cleans jagged raster edges.
    simplify_tolerance_m: float = 2.0

    # Output formats
    output_gpkg: bool = True
    output_shp: bool = True


# --------- Core Functions ---------

def extract_paths(flux, transform, crs, nodata=-9999.0,
                  config=None, release=None, dem=None, cell_counts=None):
    # Extract avalanche path polygons from a FlowPy flux raster.
    # @param flux: 2D numpy array of flux values
    # @param transform: Rasterio affine transform
    # @param crs: Coordinate reference system
    # @param nodata: NoData value in flux raster
    # @param config: PathExtractionConfig instance
    # @param release: 2D binary release area array (optional)
    # @param dem: 2D elevation array (optional, for path attributes)
    # @param cell_counts: 2D cell counts array (optional, for overlap info)
    # @returns: GeoDataFrame with path polygons and attributes

    if config is None:
        config = PathExtractionConfig()

    logger.info("Extracting avalanche path polygons from flux output...")

    # --- Step 1: Threshold flux ---
    valid = np.isfinite(flux) & (flux != nodata)
    path_mask = valid & (flux >= config.flux_threshold)

    n_path_pixels = path_mask.sum()
    if n_path_pixels == 0:
        logger.warning("No pixels above flux threshold — no paths extracted")
        return gpd.GeoDataFrame(columns=['geometry', 'path_id', 'area_m2'],
                                crs=crs)

    logger.info(f"  {n_path_pixels:,} pixels above flux threshold "
                f"({config.flux_threshold})")

    # --- Step 2: Label connected path regions ---
    path_labeled, n_paths_raw = label(path_mask)
    logger.info(f"  {n_paths_raw} connected path regions")

    # --- Step 3: Associate with release areas (optional) ---
    # Tag each path region with its contributing release area so fragments
    # from the same starting zone are merged.
    release_labels = None
    if release is not None and config.merge_by_release:
        release_labeled, n_release = label(release > 0)

        if n_release > 0:
            release_labels = _associate_paths_with_release(
                path_labeled, n_paths_raw, release_labeled
            )
            logger.info(f"  Associated paths with {n_release} release areas")

    # --- Step 4: Polygonize ---
    polygons = []
    attributes = []

    pixel_size = abs(transform[0])
    pixel_area_m2 = pixel_size * pixel_size

    for region_id in range(1, n_paths_raw + 1):
        region_mask = path_labeled == region_id
        region_area = region_mask.sum() * pixel_area_m2

        if region_area < config.min_path_area_m2:
            continue

        region_uint8 = region_mask.astype(np.uint8)
        geom_list = list(shapes(region_uint8, mask=region_mask, transform=transform))

        if not geom_list:
            continue

        # Merge all shapes for this region into one polygon
        region_shapes = [shapely_shape(g) for g, v in geom_list if v == 1]
        if not region_shapes:
            continue

        geom = unary_union(region_shapes)

        if config.simplify_tolerance_m > 0:
            geom = geom.simplify(config.simplify_tolerance_m,
                                 preserve_topology=True)

        if geom.is_empty:
            continue

        # --- Attributes ---
        attrs = {
            'path_id': region_id,
            'area_m2': float(region_area),
            'area_ha': float(region_area / 10000.0),
            'n_pixels': int(region_mask.sum()),
        }

        flux_vals = flux[region_mask]
        attrs['flux_max'] = float(flux_vals.max())
        attrs['flux_mean'] = float(flux_vals.mean())
        attrs['flux_median'] = float(np.median(flux_vals))

        if release_labels is not None and region_id in release_labels:
            attrs['release_id'] = release_labels[region_id]
        else:
            attrs['release_id'] = -1

        if dem is not None:
            dem_vals = dem[region_mask]
            valid_dem = dem_vals[np.isfinite(dem_vals)]
            if len(valid_dem) > 0:
                attrs['elev_max_m'] = float(valid_dem.max())
                attrs['elev_min_m'] = float(valid_dem.min())
                attrs['elev_drop_m'] = float(valid_dem.max() - valid_dem.min())

        if cell_counts is not None:
            cc_vals = cell_counts[region_mask]
            valid_cc = cc_vals[np.isfinite(cc_vals) & (cc_vals > 0)]
            if len(valid_cc) > 0:
                attrs['cell_count_max'] = int(valid_cc.max())
                attrs['cell_count_mean'] = float(valid_cc.mean())

        polygons.append(geom)
        attributes.append(attrs)

    if not polygons:
        logger.warning("No valid path polygons generated")
        return gpd.GeoDataFrame(columns=['geometry', 'path_id', 'area_m2'],
                                crs=crs)

    # --- Step 5: Build GeoDataFrame ---
    gdf = gpd.GeoDataFrame(attributes, geometry=polygons, crs=crs)

    if release_labels is not None and config.merge_by_release:
        gdf = _merge_by_release_area(gdf)

    logger.info(f"  Extracted {len(gdf)} path polygons")
    logger.info(f"  Total path area: {gdf['area_m2'].sum() / 1e6:.3f} km2")

    return gdf


def _associate_paths_with_release(path_labeled, n_paths, release_labeled):
    # Associate each path region with the release area it overlaps most.
    # @param path_labeled: Labeled path regions
    # @param n_paths: Number of path regions
    # @param release_labeled: Labeled release areas
    # @returns: dict mapping path_id -> release_id (-1 if no overlap)

    associations = {}

    for pid in range(1, n_paths + 1):
        path_mask = path_labeled == pid

        overlap = release_labeled[path_mask]
        overlap = overlap[overlap > 0]

        if len(overlap) > 0:
            counts = np.bincount(overlap)
            associations[pid] = int(np.argmax(counts))
        else:
            # No direct overlap (path may start just below the release area)
            associations[pid] = -1

    return associations


def _merge_by_release_area(gdf):
    # Merge path polygon fragments sharing the same release area ID.
    # @param gdf: GeoDataFrame with a release_id column
    # @returns: GeoDataFrame with merged polygons

    if 'release_id' not in gdf.columns:
        return gdf

    has_release = gdf[gdf['release_id'] > 0]
    no_release = gdf[gdf['release_id'] <= 0]

    if len(has_release) == 0:
        return gdf

    merged_rows = []

    for release_id, group in has_release.groupby('release_id'):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue

        merged_geom = unary_union(group.geometry.values)

        merged = {
            'geometry': merged_geom,
            'path_id': int(group['path_id'].iloc[0]),
            'release_id': int(release_id),
            'area_m2': float(group['area_m2'].sum()),
            'area_ha': float(group['area_m2'].sum() / 10000.0),
            'n_pixels': int(group['n_pixels'].sum()),
            'flux_max': float(group['flux_max'].max()),
            'flux_mean': float(group['flux_mean'].mean()),
            'flux_median': float(group['flux_median'].median()),
        }

        if 'elev_max_m' in group.columns:
            merged['elev_max_m'] = float(group['elev_max_m'].max())
            merged['elev_min_m'] = float(group['elev_min_m'].min())
            merged['elev_drop_m'] = float(merged['elev_max_m'] - merged['elev_min_m'])

        if 'cell_count_max' in group.columns:
            merged['cell_count_max'] = int(group['cell_count_max'].max())
            merged['cell_count_mean'] = float(group['cell_count_mean'].mean())

        merged_rows.append(merged)

    merged_gdf = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)

    result = gpd.GeoDataFrame(
        pd.concat([merged_gdf, no_release], ignore_index=True),
        crs=gdf.crs
    )

    logger.info(f"  Merged {len(has_release)} fragments -> "
                f"{len(merged_gdf)} release-associated paths "
                f"(+ {len(no_release)} unassociated)")

    return result


# --------- Runout Boundary (Single Polygon) ---------

def extract_runout_boundary(flux, transform, crs, nodata=-9999.0,
                             flux_threshold=0.003, simplify_tolerance_m=2.0):
    # Extract a single polygon for the full runout footprint: all pixels
    # above threshold unioned into one (possibly multi-part) polygon. Used
    # for the final deliverable boundary of a FlowPy run.
    # @param flux: 2D numpy array of flux values
    # @param transform: Rasterio affine transform
    # @param crs: CRS
    # @param nodata: NoData value
    # @param flux_threshold: Minimum flux for inclusion
    # @param simplify_tolerance_m: Polygon simplify tolerance (m), 0 = none
    # @returns: GeoDataFrame with a single boundary feature (may be MultiPolygon)

    logger.info("Extracting runout boundary polygon...")

    valid = np.isfinite(flux) & (flux != nodata)
    boundary_mask = valid & (flux >= flux_threshold)

    n_pixels = int(boundary_mask.sum())
    if n_pixels == 0:
        logger.warning("No pixels above flux threshold — no boundary polygon")
        return gpd.GeoDataFrame(columns=['geometry', 'area_m2'], crs=crs)

    pixel_size = abs(transform[0])
    total_area_m2 = n_pixels * pixel_size * pixel_size
    logger.info(f"  {n_pixels:,} pixels ({total_area_m2/1e6:.3f} km2) above "
                f"flux threshold {flux_threshold}")

    mask_uint8 = boundary_mask.astype(np.uint8)
    geom_list = [shapely_shape(g)
                 for g, v in shapes(mask_uint8, mask=boundary_mask, transform=transform)
                 if v == 1]

    if not geom_list:
        logger.warning("Polygonization produced no shapes")
        return gpd.GeoDataFrame(columns=['geometry', 'area_m2'], crs=crs)

    boundary = unary_union(geom_list)

    if simplify_tolerance_m > 0:
        boundary = boundary.simplify(simplify_tolerance_m, preserve_topology=True)

    if boundary.is_empty:
        logger.warning("Empty boundary after union/simplify")
        return gpd.GeoDataFrame(columns=['geometry', 'area_m2'], crs=crs)

    gdf = gpd.GeoDataFrame(
        {'geometry': [boundary], 'area_m2': [float(boundary.area)]},
        crs=crs
    )
    logger.info(f"  Boundary polygon area: {boundary.area/1e6:.3f} km2")
    return gdf


# --------- File-Based Interface ---------

def extract_paths_from_files(flux_path, output_dir, config=None,
                             release_path=None, dem_path=None,
                             cell_counts_path=None, file_stem=None):
    # Extract avalanche path polygons from FlowPy output files.
    # @param flux_path: Path to flux GeoTIFF from FlowPy
    # @param output_dir: Output directory for path polygons
    # @param config: PathExtractionConfig instance
    # @param release_path: Path to release area raster (optional)
    # @param dem_path: Path to DEM GeoTIFF (optional)
    # @param cell_counts_path: Path to cell counts raster (optional)
    # @param file_stem: Output filename stem (auto-generated if None)
    # @returns: dict with output file paths

    if config is None:
        config = PathExtractionConfig()

    flux_path = Path(flux_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_stem is None:
        file_stem = flux_path.stem.replace('_flux', '') + '_paths'

    logger.info(f"Loading flux: {flux_path.name}")
    with rasterio.open(flux_path) as src:
        flux = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else -9999.0

    release = None
    if release_path is not None:
        logger.info(f"Loading release areas: {Path(release_path).name}")
        with rasterio.open(release_path) as src:
            release = src.read(1)

    dem = None
    if dem_path is not None:
        logger.info(f"Loading DEM: {Path(dem_path).name}")
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)

    cell_counts = None
    if cell_counts_path is not None:
        logger.info(f"Loading cell counts: {Path(cell_counts_path).name}")
        with rasterio.open(cell_counts_path) as src:
            cell_counts = src.read(1).astype(np.float32)

    gdf = extract_paths(
        flux, transform, crs, nodata=nodata,
        config=config, release=release, dem=dem,
        cell_counts=cell_counts
    )

    if len(gdf) == 0:
        logger.warning("No paths extracted — no output files created")
        return {}

    outputs = {}

    if config.output_gpkg:
        gpkg_path = output_dir / f"{file_stem}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG", layer="avalanche_paths")
        outputs['paths_gpkg'] = str(gpkg_path)
        logger.info(f"Saved: {gpkg_path.name}")

    if config.output_shp:
        shp_path = output_dir / f"{file_stem}.shp"
        # Shapefile 10-char field-name limit — abbreviate the long ones
        gdf_shp = gdf.copy()
        rename_map = {
            'area_m2': 'area_m2',
            'area_ha': 'area_ha',
            'n_pixels': 'n_pixels',
            'flux_max': 'flux_max',
            'flux_mean': 'flux_mean',
            'flux_median': 'flux_med',
            'path_id': 'path_id',
            'release_id': 'rel_id',
            'elev_max_m': 'elev_max',
            'elev_min_m': 'elev_min',
            'elev_drop_m': 'elev_drop',
            'cell_count_max': 'cc_max',
            'cell_count_mean': 'cc_mean',
        }
        gdf_shp = gdf_shp.rename(columns={k: v for k, v in rename_map.items()
                                           if k in gdf_shp.columns})
        gdf_shp.to_file(shp_path)
        outputs['paths_shp'] = str(shp_path)
        logger.info(f"Saved: {shp_path.name}")

    logger.info(f"  Paths: {len(gdf)}")
    logger.info(f"  Total area: {gdf['area_m2'].sum() / 1e6:.3f} km2")
    if 'elev_drop_m' in gdf.columns:
        logger.info(f"  Elevation drop range: "
                     f"{gdf['elev_drop_m'].min():.0f} - "
                     f"{gdf['elev_drop_m'].max():.0f} m")

    # Single-polygon runout boundary (union of path pixels above threshold),
    # a hazard-extent footprint distinct from the per-path layer.
    boundary_gdf = extract_runout_boundary(
        flux, transform, crs, nodata=nodata,
        flux_threshold=config.flux_threshold,
        simplify_tolerance_m=config.simplify_tolerance_m,
    )
    if len(boundary_gdf) > 0:
        boundary_path = output_dir / f"{file_stem}_runout_boundary.gpkg"
        boundary_gdf.to_file(boundary_path, driver="GPKG", layer="runout_boundary")
        outputs['runout_boundary'] = str(boundary_path)
        logger.info(f"Saved: {boundary_path.name}")

    return outputs


# --------- CLI ---------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract avalanche path polygons from FlowPy flux output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python path_extraction.py --flux results/flux.tif --output paths/
  python path_extraction.py --flux flux.tif --release release.tif --dem dem.tif --output paths/
  python path_extraction.py --flux flux.tif --threshold 0.01 --output paths/

Flux threshold guide:
  0.003  Full extent (matches FlowPy runout limit)
  0.01   Moderate — trims marginal fringes
  0.03   Core paths only — good for visualization
  0.05   Tight core — highest confidence areas only
"""
    )

    parser.add_argument('--flux', required=True, help='Path to flux GeoTIFF from FlowPy')
    parser.add_argument('--output', required=True, help='Output directory for path polygons')
    parser.add_argument('--release', help='Path to release area raster (optional)')
    parser.add_argument('--dem', help='Path to DEM GeoTIFF (optional, for elevation attributes)')
    parser.add_argument('--cell-counts', help='Path to cell counts raster (optional)')
    parser.add_argument('--threshold', type=float, default=0.003,
                        help='Flux threshold for path delineation (default: 0.003)')
    parser.add_argument('--min-area', type=float, default=500.0,
                        help='Minimum path area in m2 (default: 500)')
    parser.add_argument('--simplify', type=float, default=2.0,
                        help='Polygon simplification tolerance in meters (default: 2.0)')
    parser.add_argument('--no-merge', action='store_true',
                        help='Do not merge path fragments by release area')
    parser.add_argument('--stem', help='Output filename stem (auto-generated if omitted)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    config = PathExtractionConfig(
        flux_threshold=args.threshold,
        min_path_area_m2=args.min_area,
        merge_by_release=not args.no_merge,
        simplify_tolerance_m=args.simplify,
    )

    outputs = extract_paths_from_files(
        flux_path=args.flux,
        output_dir=args.output,
        config=config,
        release_path=args.release,
        dem_path=args.dem,
        cell_counts_path=args.cell_counts,
        file_stem=args.stem,
    )

    if outputs:
        print(f"\nOutputs:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")
    else:
        print("\nNo paths extracted.")


if __name__ == "__main__":
    main()
