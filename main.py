"""
Main Pipeline for Statewide Avalanche Path Mapping

Author: Valerie Foley
Last Updated: 2/2026

Description:
    Three-stage pipeline over a folder of DEM tiles: Stage 1 data prep
    (terrain analysis, vegetation/FSI), Stage 2 release area detection
    (fuzzy PRA or an existing file), Stage 3 FlowPy runout modeling with
    forest friction. DEM edge effects are handled by temporarily
    mosaicking each tile with strips from its neighbors before processing,
    then clipping results back to the tile extent.
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping
from pathlib import Path
from datetime import datetime
import logging
import json
import sys
import time
import argparse
import traceback

from config import GlobalConfig, validate_config
from terrain_analysis import compute_all_terrain, save_raster, resample_dem
from vegetation_mask import create_fsi_for_dem
from release_detection import (compute_pra, pra_to_binary, filter_patches,
                                load_existing_release_areas, save_release_outputs,
                                compute_release_statistics)
from flowpy_runner import run_flowpy, create_flux_bands, FLOWPY_AVAILABLE
from path_extraction import extract_paths_from_files, PathExtractionConfig
from resource_monitor import ResourceMonitor
from adaptive_throttle import AdaptiveThrottle, set_low_priority

logger = logging.getLogger(__name__)


def _clip_release_to_aoi(release_mask, tile_name, transform, crs,
                          aoi_path, feature_id_pattern,
                          feature_idx_override=None):
    # Hard-clip release zones to the AOI feature matched by tile name. Pulls
    # an integer feature index from tile_name via feature_id_pattern (or uses
    # feature_idx_override), rasterizes that AOI polygon onto the DEM grid,
    # and zeros release labels outside it. Raises loudly on regex failure or
    # out-of-range index rather than producing silently wrong output.
    # @param release_mask: int32 label array from filter_patches
    # @param tile_name: e.g. "cluster_11"
    # @param transform: DEM affine transform
    # @param crs: DEM CRS
    # @param aoi_path: path to AOI shapefile
    # @param feature_id_pattern: regex for extracting feature index
    # @param feature_idx_override: explicit feature index, or None
    # @returns: clipped int32 label array (outside-AOI cells set to 0)

    import re
    import geopandas as gpd
    from rasterio.features import rasterize

    if feature_idx_override is not None:
        feature_idx = int(feature_idx_override)
    else:
        m = re.search(feature_id_pattern, tile_name)
        if not m:
            raise ValueError(
                f"Cannot extract feature index from DEM tile '{tile_name}' "
                f"using pattern {feature_id_pattern!r}. Either rename the "
                f"DEM to include an integer matching your AOI feature index, "
                f"or set PRAConfig.feature_id_pattern to a regex that "
                f"matches your filenames."
            )
        feature_idx = int(m.group(1))

    aoi = gpd.read_file(aoi_path)
    if feature_idx >= len(aoi) or feature_idx < 0:
        raise ValueError(
            f"Feature index {feature_idx} out of range for AOI "
            f"with {len(aoi)} features (file: {aoi_path})"
        )

    geom = aoi.iloc[feature_idx].geometry
    if aoi.crs != crs:
        geom = gpd.GeoSeries([geom], crs=aoi.crs).to_crs(crs).iloc[0]

    aoi_mask = rasterize(
        [(geom, 1)],
        out_shape=release_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    n_before = int((release_mask > 0).sum())
    clipped = np.where(aoi_mask, release_mask, 0).astype(release_mask.dtype)
    n_after = int((clipped > 0).sum())
    pct_kept = (100.0 * n_after / n_before) if n_before else 0.0
    logger.info(f"  AOI clip: feature {feature_idx} kept "
                f"{n_after:,}/{n_before:,} release pixels ({pct_kept:.1f}%)")
    return clipped


# --------- Stop-File and Resume Helpers ---------

def stop_requested(output_dir):
    # True if the supervisor asked us to stop after the current tile.
    # @param output_dir: Pipeline output directory (Path)
    # @returns: bool
    return (Path(output_dir) / "STOP_AFTER_TILE").exists()


def tile_already_done(output_dir, tile_name):
    # True if the tile's summary JSON shows status=complete. Enables resume
    # across day/night cycles without redoing tiles.
    # @param output_dir: Pipeline output directory
    # @param tile_name: tile stem
    # @returns: bool
    summary = Path(output_dir) / tile_name / f"{tile_name}_summary.json"
    if not summary.exists():
        return False
    try:
        with open(summary) as f:
            data = json.load(f)
        return data.get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


# --------- DEM Spatial Index ---------

def build_dem_index(dem_dir, dem_pattern="*.tif"):
    # Scan DEM folder and build a spatial index of tile extents.
    # @param dem_dir: Directory containing DEM tiles
    # @param dem_pattern: Glob pattern for DEM files
    # @returns: list of dicts with path, bounds, box, crs, res per DEM

    dem_dir = Path(dem_dir)
    dem_files = sorted(dem_dir.glob(dem_pattern))

    index = []
    for dem_path in dem_files:
        try:
            with rasterio.open(dem_path) as src:
                b = src.bounds
                index.append({
                    'path': dem_path,
                    'bounds': b,
                    'box': box(b.left, b.bottom, b.right, b.top),
                    'crs': src.crs,
                    'res': src.res[0]
                })
        except Exception as e:
            logger.warning(f"Could not read {dem_path.name}: {e}")

    logger.info(f"Built spatial index: {len(index)} DEMs in {dem_dir}")
    return index


def find_neighbors(tile_info, dem_index, buffer_m):
    # Find neighboring DEM tiles within buffer distance.
    # @param tile_info: dict for the primary tile (from build_dem_index)
    # @param dem_index: Full DEM spatial index
    # @param buffer_m: Buffer distance in meters
    # @returns: list of dicts for neighboring tiles (excludes self)

    search_box = tile_info['box'].buffer(buffer_m)
    neighbors = []

    for other in dem_index:
        if other['path'] == tile_info['path']:
            continue
        if search_box.intersects(other['box']):
            neighbors.append(other)

    return neighbors


def create_buffered_mosaic(tile_info, neighbors, buffer_m, output_path):
    # Build a temporary mosaic of the primary tile plus neighbor edge strips.
    # @param tile_info: Primary tile dict
    # @param neighbors: List of neighbor tile dicts
    # @param buffer_m: Buffer distance in meters
    # @param output_path: Path to write buffered mosaic
    # @returns: Path to mosaic, or original tile path if no neighbors

    if not neighbors:
        logger.info(f"  No neighbors for {tile_info['path'].name}, using original")
        return tile_info['path']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b = tile_info['bounds']
    buffered_bounds = (
        b.left - buffer_m, b.bottom - buffer_m,
        b.right + buffer_m, b.top + buffer_m
    )

    src_files = []
    all_paths = [tile_info['path']] + [n['path'] for n in neighbors]

    try:
        for p in all_paths:
            src_files.append(rasterio.open(p))

        mosaic, mosaic_transform = merge(
            src_files,
            bounds=buffered_bounds,
            nodata=-9999.0
        )

        with rasterio.open(tile_info['path']) as src:
            out_meta = src.meta.copy()

        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "compress": "LZW",
            "nodata": -9999.0
        })

        # Clean up extreme nodata values
        extreme = mosaic < -1e10
        mosaic[extreme] = -9999.0

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        logger.info(f"  Created buffered mosaic: {mosaic.shape[1]}x{mosaic.shape[2]} "
                    f"({len(neighbors)} neighbors)")
        return output_path

    finally:
        for src in src_files:
            src.close()


def clip_to_tile_extent(raster_path, tile_bounds, output_path, nodata=-9999.0):
    # Clip a raster back to the original tile extent.
    # @param raster_path: Path to raster to clip
    # @param tile_bounds: Original tile bounds (left, bottom, right, top)
    # @param output_path: Path to write clipped raster
    # @param nodata: NoData value
    # @returns: Path to clipped raster

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_geom = box(tile_bounds.left, tile_bounds.bottom,
                    tile_bounds.right, tile_bounds.top)

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio_mask(
            src, [mapping(clip_geom)], crop=True, all_touched=True, nodata=nodata
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata,
            "compress": "LZW"
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return output_path


# --------- Single Tile Processing ---------

def process_single_tile(tile_info, dem_index, config, tile_output_dir):
    # Process a single DEM tile through all three stages.
    # @param tile_info: Tile dict from build_dem_index
    # @param dem_index: Full spatial index
    # @param config: GlobalConfig instance
    # @param tile_output_dir: Output directory for this tile
    # @returns: dict with processing results

    tile_name = tile_info['path'].stem
    tile_output_dir = Path(tile_output_dir)
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing tile: {tile_name}")
    logger.info(f"{'='*60}")

    results = {
        'tile_name': tile_name,
        'tile_path': str(tile_info['path']),
        'timestamp': datetime.now().isoformat(),
        'status': 'running'
    }

    start_time = time.time()

    try:
        # ---- Smart Edge Buffering ----
        neighbors = find_neighbors(tile_info, dem_index, config.dem.dem_edge_buffer_m)
        logger.info(f"  Found {len(neighbors)} neighboring tiles")

        mosaic_path = create_buffered_mosaic(
            tile_info, neighbors, config.dem.dem_edge_buffer_m,
            tile_output_dir / f"{tile_name}_buffered.tif"
        )

        # ---- Optional DEM Resampling ----
        if config.dem.target_resolution_m is not None:
            resampled_path = tile_output_dir / f"{tile_name}_resampled.tif"
            mosaic_path = resample_dem(
                mosaic_path, resampled_path, config.dem.target_resolution_m
            )

        # ---- Stage 1: Data Prep ----
        logger.info("Stage 1: Terrain analysis...")

        wind_config = None
        if config.pra.use_wind_shelter:
            wind_config = {
                'wind_direction_deg': config.pra.wind_direction_deg,
                'search_distance_m': 300.0
            }

        terrain = compute_all_terrain(
            mosaic_path,
            smooth_sigma=config.dem.dem_smooth_sigma,
            compute_curvature=True,
            wind_shelter_config=wind_config
        )

        logger.info("Stage 1: Vegetation masking and FSI...")
        fsi_path = tile_output_dir / f"{tile_name}_fsi.tif"
        # Per-cluster canopy; falls back to config.paths.landfire_evc_path
        canopy_path = config.paths.get_canopy_for_dem(tile_info['path'])
        logger.info(f"  Canopy raster: {canopy_path.name}")
        veg_result = create_fsi_for_dem(
            mosaic_path,
            canopy_path,
            config=config.vegetation,
            output_path=str(fsi_path)
        )

        results['stage1'] = 'complete'

        # ---- Stage 2: Release Area Detection ----
        # In flowpy-only mode the existing release file is the release input.
        release_path = config.paths.get_release_area_path()

        if config.mode == "flowpy-only":
            if release_path is None:
                raise ValueError(
                    "Mode 'flowpy-only' requires a valid release_area_path "
                    "in config.paths (set via --release-areas)."
                )
            logger.info("Stage 2: SKIPPED (flowpy-only mode)")
            logger.info(f"  Using existing release areas: {release_path}")
            release_mask = load_existing_release_areas(release_path, mosaic_path)
            pra = None
            release_outputs = save_release_outputs(
                release_mask, pra,
                terrain['transform'], terrain['crs'],
                tile_output_dir / "release_areas", tile_name
            )
            results['release'] = release_outputs
            results['stage2'] = 'skipped_flowpy_only'

        else:
            # full or release-only mode
            logger.info("Stage 2: Release area detection...")

            if release_path is not None:
                # Use existing release areas even in full mode if provided
                logger.info(f"  Using existing release areas: {release_path}")
                release_mask = load_existing_release_areas(release_path, mosaic_path)
                pra = None

            else:
                # Automated fuzzy PRA detection
                pra = compute_pra(
                    terrain['slope_deg'],
                    veg_result['tree_cover_pct'],
                    config.pra,
                    mask=terrain['mask'],
                    wind_shelter=terrain.get('wind_shelter'),
                    pixel_size=terrain['pixel_size'],
                )

                release_mask = pra_to_binary(
                    pra, config.pra,
                    dem=terrain['dem'],
                    mask=terrain['mask'],
                    slope_deg=terrain['slope_deg'],
                    pixel_size=terrain['pixel_size']
                )

                # Individualize PRAs: pass 1 DEM watershed (Duvillier et al.
                # 2023), pass 2 aspect-weighted segmentation (Bühler et al. 2018)
                release_mask = filter_patches(
                    release_mask, terrain['pixel_size'], config.pra,
                    dem=terrain['dem'], mask=terrain['mask'],
                    slope_deg=terrain['slope_deg'],
                    aspect_deg=terrain['aspect_deg']
                )

                # Optional hard-clip to AOI feature footprint (FlowPy keeps
                # the full buffered DEM for runout).
                if config.pra.clip_to_aoi:
                    # tile_processor sets this so all sub-tiles of a cluster
                    # point at the same feature.
                    override = getattr(config.pra, "_feature_idx_override", None)
                    release_mask = _clip_release_to_aoi(
                        release_mask=release_mask,
                        tile_name=tile_name,
                        transform=terrain['transform'],
                        crs=terrain['crs'],
                        aoi_path=config.paths.aoi_clip_path,
                        feature_id_pattern=config.pra.feature_id_pattern,
                        feature_idx_override=override,
                    )

                # Optional core-bbox clip: tile_processor drops the overlap
                # so adjacent tiles don't double up.
                core_bbox = getattr(config.pra, "_core_bbox_override", None)
                if core_bbox is not None:
                    from release_detection import clip_labels_to_bbox
                    release_mask = clip_labels_to_bbox(
                        release_mask, terrain['transform'], core_bbox
                    )

            release_outputs = save_release_outputs(
                release_mask, pra,
                terrain['transform'], terrain['crs'],
                tile_output_dir / "release_areas", tile_name
            )
            results['release'] = release_outputs

            # compute_release_statistics expects boolean
            stats = compute_release_statistics(
                release_mask > 0, terrain['slope_deg'], terrain['dem'],
                terrain['transform'], terrain['pixel_size']
            )
            results['statistics'] = stats
            logger.info(f"  Release area: {stats['release_area_km2']:.3f} km2, "
                        f"{stats['patches']['count']} patches")

            results['stage2'] = 'complete'

        # ---- Stage 3: FlowPy Runout ----
        # Skipped in release-only mode, if FlowPy isn't installed, or if
        # there are no release pixels to seed from.
        if not config.run_flowpy():
            logger.info("Stage 3: SKIPPED (release-only mode)")
            results['stage3'] = 'skipped_release_only'
        elif not FLOWPY_AVAILABLE:
            logger.warning("FlowPy not available - skipping Stage 3")
            results['stage3'] = 'skipped'
        elif (release_mask > 0).sum() == 0:
            logger.warning("No release areas detected - skipping FlowPy")
            results['stage3'] = 'skipped_no_release'
        else:
            logger.info("Stage 3: FlowPy runout modeling...")

            release_raster = release_outputs.get('release_raster')

            flowpy_outputs = run_flowpy(
                dem_path=mosaic_path,
                release_path=release_raster,
                output_dir=tile_output_dir / "flowpy",
                flowpy_config=config.flowpy,
                fsi_path=str(fsi_path) if config.flowpy.use_forest else None,
                run_id=tile_name
            )

            if flowpy_outputs.get('flux'):
                flux_bands = create_flux_bands(
                    flowpy_outputs['flux'],
                    tile_output_dir / "flowpy" / f"{tile_name}_flux_bands.tif"
                )
                flowpy_outputs['flux_bands'] = flux_bands

            # Extract path polygons + runout boundary from flux
            if flowpy_outputs.get('flux'):
                logger.info("Stage 3b: Extracting path polygons from flux...")
                path_config = PathExtractionConfig()
                path_outputs = extract_paths_from_files(
                    flux_path=flowpy_outputs['flux'],
                    output_dir=tile_output_dir / "paths",
                    config=path_config,
                    release_path=release_outputs.get('release_raster'),
                    dem_path=mosaic_path,
                    cell_counts_path=flowpy_outputs.get('cellCounts'),
                    file_stem=tile_name
                )
                results['paths'] = path_outputs

            results['flowpy'] = flowpy_outputs
            results['stage3'] = 'complete'

        # ---- Clip results back to tile extent (remove buffer zone) ----
        if neighbors:
            logger.info("Clipping results to original tile extent...")
            for key in ['release_raster', 'pra_raster']:
                if key in release_outputs and release_outputs[key]:
                    clipped = clip_to_tile_extent(
                        release_outputs[key],
                        tile_info['bounds'],
                        Path(release_outputs[key]).parent / f"{tile_name}_{key}_clipped.tif"
                    )
                    release_outputs[f'{key}_clipped'] = str(clipped)

            if results.get('stage3') == 'complete' and results.get('flowpy'):
                for key, path in results['flowpy'].items():
                    if path and Path(path).exists() and path.endswith('.tif'):
                        clipped = clip_to_tile_extent(
                            path,
                            tile_info['bounds'],
                            Path(path).parent / f"{Path(path).stem}_clipped.tif"
                        )
                        results['flowpy'][f'{key}_clipped'] = str(clipped)

            # Clean up temporary mosaic / resampled DEM
            mosaic_file = tile_output_dir / f"{tile_name}_buffered.tif"
            if mosaic_file.exists():
                mosaic_file.unlink()
                logger.info("  Removed temporary mosaic")

            resampled_file = tile_output_dir / f"{tile_name}_resampled.tif"
            if resampled_file.exists():
                resampled_file.unlink()
                logger.info("  Removed temporary resampled DEM")

        duration = time.time() - start_time
        results['status'] = 'complete'
        results['duration_seconds'] = duration
        logger.info(f"Tile complete: {tile_name} ({duration:.1f}s)")

    except Exception as e:
        duration = time.time() - start_time
        results['status'] = 'failed'
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
        results['duration_seconds'] = duration
        logger.error(f"Tile failed: {tile_name}: {e}")
        logger.error(traceback.format_exc())

    summary_path = tile_output_dir / f"{tile_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# --------- Max Composite ---------

def max_composite_outputs(tile_results, output_dir, output_type='flux'):
    # Create a per-pixel max composite of overlapping tile outputs.
    # @param tile_results: List of tile result dicts
    # @param output_dir: Output directory for composite
    # @param output_type: Output to composite (e.g. 'flux', 'z_delta')
    # @returns: Path to composite raster, or None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect clipped (or fallback regular) output paths
    paths = []
    for result in tile_results:
        if result.get('status') != 'complete':
            continue
        flowpy = result.get('flowpy', {})
        clipped_key = f'{output_type}_clipped'
        path = flowpy.get(clipped_key) or flowpy.get(output_type)
        if path and Path(path).exists():
            paths.append(Path(path))

    if not paths:
        logger.warning(f"No {output_type} outputs to composite")
        return None

    if len(paths) == 1:
        logger.info(f"Only 1 tile for {output_type}, no composite needed")
        return str(paths[0])

    logger.info(f"Creating max composite of {len(paths)} {output_type} tiles...")

    src_files = [rasterio.open(p) for p in paths]
    try:
        mosaic, mosaic_transform = merge(src_files, method='max', nodata=-9999.0)

        with rasterio.open(paths[0]) as ref:
            out_meta = ref.meta.copy()

        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "compress": "LZW",
            "nodata": -9999.0
        })

        composite_path = output_dir / f"composite_{output_type}.tif"
        with rasterio.open(composite_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        logger.info(f"Saved composite: {composite_path.name}")
        return str(composite_path)

    finally:
        for src in src_files:
            src.close()


# --------- Main Pipeline ---------

def run_pipeline(config, monitor=None, throttle=None):
    # Run the full pipeline across all DEM tiles.
    # @param config: GlobalConfig instance
    # @param monitor: ResourceMonitor instance, or None
    # @param throttle: AdaptiveThrottle instance, or None
    # @returns: list of tile result dicts

    logger.info(config.summary())

    warnings = validate_config(config)
    if warnings:
        for w in warnings:
            logger.warning(f"Config warning: {w}")

    dem_index = build_dem_index(config.paths.dem_dir, config.paths.dem_pattern)

    if not dem_index:
        logger.error("No DEM files found!")
        return []

    # Pre-scan: count tiles already complete from previous runs
    n_skipped = sum(
        1 for t in dem_index
        if tile_already_done(config.paths.output_dir, t['path'].stem)
    )
    if n_skipped:
        logger.info(f"Resume: {n_skipped}/{len(dem_index)} tiles already complete, "
                    f"will be skipped")

    # Sequential processing (parallel via multiprocessing.Pool can be added later)
    all_results = []
    total = len(dem_index)
    stopped_early = False

    for i, tile_info in enumerate(dem_index, 1):
        tile_name = tile_info['path'].stem

        if tile_already_done(config.paths.output_dir, tile_name):
            logger.info(f"Tile {i}/{total}: {tile_name} already complete, skipping")
            continue

        # Honor supervisor stop request BEFORE starting a new tile
        if stop_requested(config.paths.output_dir):
            logger.info(f"\nStop requested by supervisor (STOP_AFTER_TILE present). "
                        f"Exiting after {i-1} tiles processed this session, "
                        f"{total - i + 1} remaining.")
            stopped_early = True
            break

        # Wait for resources if busy, and adjust FlowPy CPU dynamically
        if throttle is not None:
            state = throttle.wait_until_ok()
            new_cap = throttle.recommended_cpu_cap(state)
            if new_cap != config.flowpy.cpu_cap:
                logger.info(f"Throttle adjusting FlowPy cpu_cap: "
                            f"{config.flowpy.cpu_cap} -> {new_cap}")
                config.flowpy.cpu_cap = new_cap
            logger.info(f"Throttle status: {throttle.status_line(state)}")

        logger.info(f"\nTile {i}/{total}")
        tile_output_dir = config.paths.output_dir / tile_name

        if monitor is not None:
            with rasterio.open(tile_info['path']) as src:
                n_pixels = src.width * src.height
            monitor.tile_start(tile_name, dem_pixels=n_pixels)

        result = process_single_tile(tile_info, dem_index, config, tile_output_dir)
        all_results.append(result)

        if monitor is not None:
            monitor.tile_end(
                status=result.get('status', 'unknown'),
                error=result.get('error', '')
            )

        if i % config.performance.checkpoint_interval == 0:
            checkpoint_path = config.paths.output_dir / f"checkpoint_{i}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Build composites only on a full session. On a windowed early-stop we
    # exit cleanly so the supervisor resumes tomorrow; composites wait.
    if stopped_early:
        logger.info("\nSkipping composites (will be built when all tiles done)")
    else:
        logger.info("\nCreating output composites...")
        composite_dir = config.paths.output_dir / "composites"

        for output_type in ['flux', 'z_delta', 'z_delta_sum', 'cell_counts']:
            max_composite_outputs(all_results, composite_dir, output_type)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tiles': total,
        'successful': sum(1 for r in all_results if r['status'] == 'complete'),
        'failed': sum(1 for r in all_results if r['status'] == 'failed'),
        'tile_results': all_results
    }

    summary_path = config.paths.output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Tiles processed: {total}")
    logger.info(f"  Successful: {summary['successful']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Results: {config.paths.output_dir}")
    logger.info(f"{'='*60}")

    return all_results


# --------- CLI ---------

def parse_args():
    # Parse command-line arguments.
    # @returns: argparse.Namespace

    parser = argparse.ArgumentParser(
        description='Statewide Avalanche Path Mapping Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --verbose
  python main.py --dem-dir /path/to/dems --output-dir /path/to/outputs
  python main.py --workers 4
"""
    )

    parser.add_argument('--dem-dir', type=str, help='Override DEM directory')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--mode', type=str,
                        choices=['full', 'release-only', 'flowpy-only'],
                        default=None,
                        help='Pipeline mode: full (default), release-only '
                             '(skip FlowPy), or flowpy-only (skip PRA, '
                             'requires --release-areas).')
    parser.add_argument('--evc-path', type=str, help='Override LANDFIRE EVC path')
    parser.add_argument('--canopy-dir', type=str,
                        help='Directory of per-cluster canopy rasters '
                             '(matched by DEM tile basename)')
    parser.add_argument('--aoi-clip-path', type=str,
                        help='Path to AOI shapefile for hard-clipping '
                             'release zones (only used if --clip-to-aoi)')
    parser.add_argument('--clip-to-aoi', dest='clip_to_aoi',
                        action='store_true', default=None,
                        help='Hard-clip release zones to AOI feature footprint')
    parser.add_argument('--no-clip-to-aoi', dest='clip_to_aoi',
                        action='store_false',
                        help='Disable AOI hard-clipping (release everywhere)')
    parser.add_argument('--release-areas', type=str, help='Path to existing release areas')

    parser.add_argument('--workers', type=int, help='Override worker count')
    parser.add_argument('--resolution', type=float,
                        help='Target DEM resolution in meters (e.g. 5 or 10). '
                             'Resamples from native before processing.')
    parser.add_argument('--edge-buffer', type=float, help='DEM edge buffer in meters')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable resource monitoring (CPU, memory, disk per tile)')
    parser.add_argument('--monitor-interval', type=float, default=5.0,
                        help='Resource monitor sampling interval in seconds')
    parser.add_argument('--throttle', action='store_true',
                        help='Enable adaptive throttling for shared servers '
                             '(waits between tiles when load is high, scales '
                             'FlowPy CPU dynamically). Recommended for '
                             'production runs on shared boxes.')
    parser.add_argument('--low-priority', action='store_true',
                        help='Set process to lowest CPU and I/O priority '
                             '(nice +19, ionice idle). Recommended for '
                             'shared servers.')
    parser.add_argument('--target-cpu-fraction', type=float, default=0.25,
                        help='Fraction of total CPUs to target when throttling '
                             '(default 0.25 = polite share)')
    parser.add_argument('--target-mem-fraction', type=float, default=0.25,
                        help='Fraction of total RAM to target when throttling '
                             '(default 0.25)')
    parser.add_argument('--min-free-mem-gb', type=float, default=32.0,
                        help='Pause if available RAM drops below this (default 32 GB)')
    parser.add_argument('--min-free-disk-gb', type=float, default=50.0,
                        help='Pause if free disk drops below this (default 50 GB)')

    return parser.parse_args()


def main():
    # Main entry point (args from command line).

    args = parse_args()

    config = GlobalConfig()

    # Apply command-line overrides
    if args.mode:
        config.set_mode(args.mode)
    if args.dem_dir:
        config.paths.dem_dir = Path(args.dem_dir)
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
    if args.evc_path:
        config.paths.landfire_evc_path = Path(args.evc_path)
    if args.canopy_dir:
        config.paths.canopy_dir = Path(args.canopy_dir)
    if args.aoi_clip_path:
        config.paths.aoi_clip_path = Path(args.aoi_clip_path)
    if args.clip_to_aoi is not None:
        config.pra.clip_to_aoi = args.clip_to_aoi
    if args.release_areas:
        config.paths.release_area_path = args.release_areas
    if args.workers:
        config.performance.max_workers = args.workers
    if args.resolution is not None:
        config.dem.target_resolution_m = args.resolution
    if args.edge_buffer is not None:
        config.dem.dem_edge_buffer_m = args.edge_buffer

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(config.paths.log_dir / "pipeline.log"),
            logging.StreamHandler()
        ]
    )

    try:
        # Lower process priority before doing any work
        if args.low_priority:
            set_low_priority()

        monitor = None
        if args.monitor:
            monitor = ResourceMonitor(
                config.paths.log_dir,
                interval_seconds=args.monitor_interval
            )
            monitor.start()

        throttle = None
        if args.throttle:
            throttle = AdaptiveThrottle(
                target_cpu_fraction=args.target_cpu_fraction,
                target_mem_fraction=args.target_mem_fraction,
                min_free_mem_gb=args.min_free_mem_gb,
                min_free_disk_gb=args.min_free_disk_gb,
                flowpy_cpu_floor=2,
                flowpy_cpu_ceiling=config.flowpy.cpu_cap,
                output_disk=str(config.paths.output_dir),
            )
            logger.info(f"Adaptive throttle enabled: "
                        f"{throttle.status_line(throttle.last_state)}")

        run_pipeline(config, monitor=monitor, throttle=throttle)

        if monitor is not None:
            monitor.stop()

    except Exception as e:
        if monitor is not None:
            monitor.stop()
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
