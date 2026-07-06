"""
Tile Runner

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Wraps process_single_tile from main.py so it can run on a cropped
    sub-tile, building the minimal tile_info structure it expects and
    routing outputs to outputs/<cluster_id>/tiles/tile_<row>_<col>/.
    Passes the tile's core_bbox down via config.pra._core_bbox_override so
    release labels are clipped to the core AFTER segmentation - the overlap
    then only feeds neighborhood operations (watershed, aspect segmentation)
    and isn't double-counted in the mosaic, which removes the tile-seam
    artifact. FlowPy runout outputs are left unclipped (they need the
    overlap; the mosaicker's max-merge handles it). Mosaicking itself lives
    in raster_mosaicker.py; this module only runs one tile and reports.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict

import rasterio
from shapely.geometry import box

from .tile_grid import Tile

logger = logging.getLogger(__name__)


# --------- Output Layout ---------

def tile_output_dir(cluster_output_dir: Path, tile: Tile) -> Path:
    # Where this single sub-tile's outputs live.
    # @param cluster_output_dir: parent output dir for the cluster
    # @param tile: Tile
    # @returns: e.g. outputs/cluster_11/tiles/tile_00_00/
    return Path(cluster_output_dir) / "tiles" / f"tile_{tile.tile_id}"


# --------- Tile Info Construction ---------

def _build_tile_info(cropped_dem_path: Path) -> Dict:
    # Build the tile_info dict process_single_tile expects. The cropped DEM
    # IS the tile; overlap is already baked in by the cropping step, so no
    # neighbors are needed.
    # @param cropped_dem_path: per-tile DEM
    # @returns: {path, bounds, box, crs, res} matching build_dem_index

    with rasterio.open(cropped_dem_path) as src:
        b = src.bounds
        return {
            "path": Path(cropped_dem_path),
            "bounds": b,
            "box": box(b.left, b.bottom, b.right, b.top),
            "crs": src.crs,
            "res": src.res[0],
        }


# --------- Canopy Override ---------

def _override_canopy_for_tile(config, cropped_canopy_path: Path):
    # Point get_canopy_for_dem at our per-tile crop instead of the
    # cluster-wide canopy. It matches a canopy file to the DEM basename, but
    # our per-tile DEM ("tile_00_00_dem.tif") and canopy ("..._canopy.tif")
    # differ, so that match fails. Bypass: set landfire_evc_path to the
    # per-tile canopy and clear canopy_dir so it falls back to that path.
    # @param config: GlobalConfig
    # @param cropped_canopy_path: per-tile canopy
    # @returns: restore callback

    saved_canopy_dir = config.paths.canopy_dir
    saved_evc_path = config.paths.landfire_evc_path

    config.paths.canopy_dir = Path("")
    config.paths.landfire_evc_path = Path(cropped_canopy_path)

    def restore():
        config.paths.canopy_dir = saved_canopy_dir
        config.paths.landfire_evc_path = saved_evc_path

    return restore


# --------- Skip-If-Done ---------

def is_tile_done(cluster_output_dir: Path, tile: Tile) -> bool:
    # True if the tile's pipeline_summary.json exists and reports complete.
    # Used to skip already-processed tiles on resume.
    # @param cluster_output_dir: parent output dir for the cluster
    # @param tile: Tile
    # @returns: bool
    summary = tile_output_dir(cluster_output_dir, tile) / "pipeline_summary.json"
    if not summary.exists():
        return False
    try:
        import json
        with open(summary) as f:
            data = json.load(f)
        return data.get("status") in ("complete", "completed")
    except Exception:
        return False


# --------- Main Entry Point ---------

def run_tile(tile: Tile,
             cluster_id: str,
             cropped_dem_path: Path,
             cropped_canopy_path: Path,
             config,
             cluster_output_dir: Path,
             skip_if_done: bool = True) -> Dict:
    # Run the per-tile pipeline (process_single_tile) on a cropped sub-tile.
    # @param tile: Tile from compute_tile_grid()
    # @param cluster_id: e.g. "cluster_11"
    # @param cropped_dem_path: per-tile DEM from tile_input_cropping
    # @param cropped_canopy_path: per-tile canopy from tile_input_cropping
    # @param config: GlobalConfig
    # @param cluster_output_dir: parent output dir for this cluster
    # @param skip_if_done: if True, return cached result when summary exists
    # @returns: dict with status, tile_id, output_dir, elapsed_s, [error]

    out_dir = tile_output_dir(cluster_output_dir, tile)

    if skip_if_done and is_tile_done(cluster_output_dir, tile):
        logger.info(f"  tile {tile.tile_id}: SKIP (already complete)")
        return {
            "status": "skipped",
            "tile_id": tile.tile_id,
            "output_dir": str(out_dir),
            "elapsed_s": 0.0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  tile {tile.tile_id}: starting -> {out_dir}")

    # Lazy import to avoid circular import at module load time
    from main import process_single_tile

    tile_info = _build_tile_info(cropped_dem_path)

    # Single-entry dem_index: find_neighbors sees the tile, skips itself,
    # returns [] neighbors, and create_buffered_mosaic just copies the tile.
    dem_index = [tile_info]

    # Use our per-tile canopy crop
    restore_canopy = _override_canopy_for_tile(config, cropped_canopy_path)

    # AOI feature index must come from cluster_id, not the sub-tile name
    # (whose leading integer like "00" is a tile coordinate, not a feature).
    import re as _re
    cluster_match = _re.search(r"(\d+)", cluster_id)
    if cluster_match is None:
        raise ValueError(
            f"cluster_id {cluster_id!r} contains no integer for AOI lookup"
        )
    saved_aoi_override = getattr(config.pra, "_feature_idx_override", None)
    config.pra._feature_idx_override = int(cluster_match.group(1))

    # core_bbox clips release labels to the tile core after segmentation -
    # fixes the seam between adjacent tiles in past mosaics.
    saved_core_bbox = getattr(config.pra, "_core_bbox_override", None)
    config.pra._core_bbox_override = tuple(tile.core_bbox)

    def restore_overrides():
        if saved_aoi_override is None:
            try:
                del config.pra._feature_idx_override
            except AttributeError:
                pass
        else:
            config.pra._feature_idx_override = saved_aoi_override

        if saved_core_bbox is None:
            try:
                del config.pra._core_bbox_override
            except AttributeError:
                pass
        else:
            config.pra._core_bbox_override = saved_core_bbox

    t0 = time.time()
    try:
        results = process_single_tile(
            tile_info=tile_info,
            dem_index=dem_index,
            config=config,
            tile_output_dir=out_dir,
        )
        elapsed = time.time() - t0

        status = results.get("status", "unknown")
        logger.info(f"  tile {tile.tile_id}: done in {elapsed/60:.1f} min "
                    f"(status={status})")

        return {
            "status": status,
            "tile_id": tile.tile_id,
            "output_dir": str(out_dir),
            "elapsed_s": elapsed,
            "results": results,
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  tile {tile.tile_id}: FAILED after {elapsed/60:.1f} min: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "status": "failed",
            "tile_id": tile.tile_id,
            "output_dir": str(out_dir),
            "elapsed_s": elapsed,
            "error": str(e),
        }
    finally:
        restore_canopy()
        restore_overrides()


# --------- Smoke Test ---------

if __name__ == "__main__":
    import argparse
    from .tile_grid import compute_tile_grid
    from .tile_input_cropping import crop_tile_inputs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import GlobalConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [tile-runner] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run pipeline on one sub-tile")
    parser.add_argument("--cluster-id", default="cluster_11")
    parser.add_argument("--tile-id", default="00_00",
                        help="Tile id like '00_00'")
    parser.add_argument("--dem-dir", default="/home/val/regionalRunout/data/DEM")
    parser.add_argument("--canopy-dir", default="/home/val/regionalRunout/data/forests")
    parser.add_argument("--work-dir", default="/home/val/regionalRunout/work")
    parser.add_argument("--output-dir", default="/home/val/regionalRunout/outputs")
    parser.add_argument("--mode", default=None,
                        choices=['full', 'release-only', 'flowpy-only'],
                        help="Pipeline mode (overrides config default)")
    parser.add_argument("--release-areas", default=None,
                        help="Path to release polygon/raster file. Required "
                             "for --mode flowpy-only. Used as input releases "
                             "(not auto-detected via fuzzy logic) in any mode "
                             "when provided.")
    args = parser.parse_args()

    cluster_id = args.cluster_id
    target_tile_id = args.tile_id

    dem_path = Path(args.dem_dir) / f"{cluster_id}.tif"
    canopy_path = Path(args.canopy_dir) / f"{cluster_id}.tif"
    work_dir = Path(args.work_dir)
    cluster_out_dir = Path(args.output_dir) / cluster_id

    with rasterio.open(dem_path) as src:
        bbox = (src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top)
    tiles = compute_tile_grid(bbox, core_m=9000.0, overlap_m=3000.0)

    target = next((t for t in tiles if t.tile_id == target_tile_id), None)
    if target is None:
        print(f"ERROR: tile_id {target_tile_id} not found in grid")
        print(f"Available tile ids: {[t.tile_id for t in tiles]}")
        sys.exit(1)

    print(f"Cluster:      {cluster_id}")
    print(f"Tile:         {target_tile_id}")
    print(f"Output dir:   {tile_output_dir(cluster_out_dir, target)}")
    print()

    print(">> Cropping inputs...")
    paths = crop_tile_inputs(target, cluster_id, dem_path, canopy_path, work_dir)
    print(f"   DEM:    {paths['dem']}")
    print(f"   Canopy: {paths['canopy']}")
    print()

    print(">> Running pipeline (this may take 10-60 minutes)...")
    config = GlobalConfig()
    if args.mode:
        config.set_mode(args.mode)
    if args.release_areas:
        config.paths.release_area_path = str(args.release_areas)
    result = run_tile(
        tile=target,
        cluster_id=cluster_id,
        cropped_dem_path=paths["dem"],
        cropped_canopy_path=paths["canopy"],
        config=config,
        cluster_output_dir=cluster_out_dir,
        skip_if_done=False,
    )

    print()
    print(f"Result: {result['status']} in {result['elapsed_s']/60:.1f} min")
    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Outputs in: {result['output_dir']}")
