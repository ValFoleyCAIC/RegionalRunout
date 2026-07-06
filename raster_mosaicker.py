"""
Per-Cluster Raster Mosaicker

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Combines per-tile output rasters from a tiled cluster run into one
    cluster-level raster. Flux-like outputs (flux, zDelta, zDeltaSum,
    cellCounts, travelLength) are merged with per-pixel max(): each output
    is "max possible from any upstream release", so where tiles overlap the
    most-informed value wins and single-tile pixels keep their value. Source
    nodata is treated as no contribution, so only valid pixels take part in
    the max. Output extent is the union of tile cores.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import rasterio
from rasterio.merge import merge

from .tile_grid import Tile

logger = logging.getLogger(__name__)


# --------- Tile Output Discovery ---------

def discover_tile_output(tile_dir: Path, output_type: str) -> Optional[Path]:
    # Find the FlowPy output raster for a given type in a tile's output dir.
    # FlowPy names outputs com4_<tile_name>_<timestamp>_<type>.tif; match
    # case-insensitively on the _<type> suffix, with a substring fallback.
    # @param tile_dir: per-tile output directory
    # @param output_type: e.g. "flux", "zdelta", "cellCounts"
    # @returns: Path to the matching tif, or None

    flowpy_results = Path(tile_dir) / "flowpy" / "results"
    if not flowpy_results.exists():
        return None

    type_lower = output_type.lower()
    candidates = []
    for tif in flowpy_results.glob("*.tif"):
        if tif.stem.lower().endswith(f"_{type_lower}"):
            candidates.append(tif)

    if not candidates:
        # Looser substring match
        for tif in flowpy_results.glob("*.tif"):
            if type_lower in tif.stem.lower():
                candidates.append(tif)

    if not candidates:
        return None
    if len(candidates) > 1:
        # Newest by mtime (most recent FlowPy run)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# --------- Main Mosaicking ---------

def mosaic_one_output_type(cluster_id: str,
                            tiles: List[Tile],
                            cluster_output_dir: Path,
                            output_type: str,
                            final_output_path: Path) -> Optional[Path]:
    # Mosaic all per-tile rasters of one output_type into final_output_path
    # using method='max' (valid value wins where a pixel is nodata in some
    # sources and valid in others).
    # @param cluster_id: e.g. "cluster_11"
    # @param tiles: cluster tile grid
    # @param cluster_output_dir: e.g. outputs/cluster_11/
    # @param output_type: FlowPy output name
    # @param final_output_path: mosaic output path
    # @returns: final_output_path, or None if no tile inputs were found

    from .tile_runner import tile_output_dir

    tile_rasters = []
    missing = []
    for t in tiles:
        td = tile_output_dir(cluster_output_dir, t)
        out = discover_tile_output(td, output_type)
        if out is None:
            missing.append(t.tile_id)
        else:
            tile_rasters.append(out)

    if missing:
        logger.warning(f"  {output_type}: missing in {len(missing)} tile(s): "
                       f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not tile_rasters:
        logger.warning(f"  {output_type}: no inputs found, skipping")
        return None

    logger.info(f"  {output_type}: merging {len(tile_rasters)} tile rasters "
                f"-> {final_output_path.name}")

    src_files = []
    try:
        for path in tile_rasters:
            src_files.append(rasterio.open(path))

        nodata = src_files[0].nodata if src_files[0].nodata is not None else -9999.0

        mosaic, mosaic_transform = merge(
            src_files,
            method='max',
            nodata=nodata,
        )

        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "compress": "LZW",
            "tiled": True,
            "nodata": nodata,
        })

        final_output_path = Path(final_output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(final_output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        return final_output_path
    finally:
        for s in src_files:
            s.close()


def mosaic_tile_outputs(cluster_id: str,
                         tiles: List[Tile],
                         cluster_output_dir: Path,
                         output_types: Optional[List[str]] = None) -> Dict[str, Optional[Path]]:
    # Mosaic all output types for one cluster.
    # @param cluster_id: e.g. "cluster_11"
    # @param tiles: cluster tile grid
    # @param cluster_output_dir: e.g. outputs/cluster_11/
    # @param output_types: FlowPy output names (default: standard 5)
    # @returns: dict mapping output_type -> final mosaic path (or None)

    if output_types is None:
        output_types = ["flux", "zdelta", "cellCounts", "zDeltaSum", "travelLength"]

    cluster_output_dir = Path(cluster_output_dir)
    results = {}

    logger.info(f"Mosaicking cluster {cluster_id}: {len(tiles)} tiles, "
                f"{len(output_types)} output types")

    for ot in output_types:
        out_path = cluster_output_dir / f"{cluster_id}_{ot}.tif"
        try:
            result = mosaic_one_output_type(
                cluster_id, tiles, cluster_output_dir, ot, out_path
            )
            results[ot] = result
        except Exception as e:
            logger.error(f"  {ot}: failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results[ot] = None

    succeeded = sum(1 for v in results.values() if v is not None)
    logger.info(f"Mosaicked {succeeded}/{len(output_types)} output types for {cluster_id}")
    return results


# --------- Smoke Test ---------

if __name__ == "__main__":
    import argparse
    from .tile_grid import compute_tile_grid

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [mosaic] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Mosaic tile outputs into final cluster rasters")
    parser.add_argument("--cluster-id", default="cluster_11")
    parser.add_argument("--dem-dir", default="/home/val/regionalRunout/data/DEM")
    parser.add_argument("--output-dir", default="/home/val/regionalRunout/outputs")
    args = parser.parse_args()

    cluster_id = args.cluster_id
    dem_path = Path(args.dem_dir) / f"{cluster_id}.tif"
    cluster_out = Path(args.output_dir) / cluster_id

    # Reconstruct the same tile grid used for processing
    with rasterio.open(dem_path) as src:
        bbox = (src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top)
    tiles = compute_tile_grid(bbox, core_m=9000.0, overlap_m=3000.0)

    print(f"Cluster:     {cluster_id}")
    print(f"Tile count:  {len(tiles)}")
    print(f"Output dir:  {cluster_out}")
    print()

    results = mosaic_tile_outputs(cluster_id, tiles, cluster_out)

    print()
    print("Results:")
    for ot, path in results.items():
        if path:
            with rasterio.open(path) as src:
                size = path.stat().st_size / (1024**2)
                print(f"  {ot:20s} {src.width}x{src.height}  {size:.0f} MB  -> {path.name}")
        else:
            print(f"  {ot:20s} (no output)")
