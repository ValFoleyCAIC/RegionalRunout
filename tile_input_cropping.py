"""
Tile Input Cropping

Author: Valerie Foley
Last Updated: 5/2026

Description:
    For a single tile of a cluster, extracts the tile's full_bbox region
    from the cluster's DEM and canopy raster and writes per-tile inputs to
    work_dir/tiles/<cluster_id>/ (tile_<row>_<col>_dem.tif and _canopy.tif),
    ready for the per-tile pipeline. DEM is written in the source CRS with
    pixel alignment preserved (windowed read snaps to the source grid);
    edge tiles extending past the DEM are padded with nodata. Canopy is
    cropped and, if needed, reprojected to match the per-tile DEM grid.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from .tile_grid import Tile

logger = logging.getLogger(__name__)


# --------- Helpers ---------

def _tile_dirname(work_dir: Path, cluster_id: str) -> Path:
    # Per-cluster scratch directory for tile inputs.
    return Path(work_dir) / "tiles" / cluster_id


def _tile_filename(tile: Tile, kind: str, ext: str = "tif") -> str:
    # e.g. "tile_03_05_dem.tif"
    return f"tile_{tile.tile_id}_{kind}.{ext}"


def _bbox_to_pixel_aligned_window(bbox, src_transform, src_width, src_height):
    # Convert a bbox in source CRS to a pixel-aligned Window, snapped to the
    # source grid so the crop needs no resampling. The returned window may
    # extend past the source bounds (negative offsets or oversize); the
    # caller handles padding/clipping.
    # @param bbox: (minx, miny, maxx, maxy) in source CRS
    # @param src_transform: source affine transform
    # @param src_width: source raster width (px)
    # @param src_height: source raster height (px)
    # @returns: (window, dst_transform, dst_width, dst_height)

    minx, miny, maxx, maxy = bbox
    src_minx, src_maxy = src_transform.c, src_transform.f
    px_w = src_transform.a       # pixel width  (positive)
    px_h = -src_transform.e      # pixel height (positive; src e is negative)

    # floor the top-left offset so the window covers the bbox
    col_off_f = (minx - src_minx) / px_w
    row_off_f = (src_maxy - maxy) / px_h
    col_off = int(np.floor(col_off_f))
    row_off = int(np.floor(row_off_f))

    # ceil the extent so the window fully spans the bbox
    width_f = (maxx - minx) / px_w
    height_f = (maxy - miny) / px_h
    width  = int(np.ceil(col_off_f + width_f) - col_off)
    height = int(np.ceil(row_off_f + height_f) - row_off)

    window = Window(col_off, row_off, width, height)

    # World bbox of the snapped window
    snap_minx = src_minx + col_off * px_w
    snap_maxy = src_maxy - row_off * px_h
    dst_transform = from_origin(snap_minx, snap_maxy, px_w, px_h)

    return window, dst_transform, width, height


def _read_window_with_padding(src, window, fill_value):
    # Read a window that may extend past src's extent, filling cells outside
    # the real extent with fill_value.
    # @param src: open rasterio dataset
    # @param window: Window (may exceed src bounds)
    # @param fill_value: value for out-of-extent cells
    # @returns: array of shape (window.height, window.width)

    inter_col_off = max(window.col_off, 0)
    inter_row_off = max(window.row_off, 0)
    inter_col_end = min(window.col_off + window.width, src.width)
    inter_row_end = min(window.row_off + window.height, src.height)
    inter_w = inter_col_end - inter_col_off
    inter_h = inter_row_end - inter_row_off

    out = np.full(
        (window.height, window.width),
        fill_value,
        dtype=src.dtypes[0] if src.dtypes[0] != "uint8" else np.float32,
    )

    if inter_w > 0 and inter_h > 0:
        inter_window = Window(inter_col_off, inter_row_off, inter_w, inter_h)
        data = src.read(1, window=inter_window)

        dst_col = inter_col_off - window.col_off
        dst_row = inter_row_off - window.row_off
        out[dst_row:dst_row + inter_h, dst_col:dst_col + inter_w] = data

    return out


# --------- DEM Cropping ---------

def crop_dem_to_tile(tile: Tile, dem_path: Path, output_path: Path) -> Path:
    # Crop a DEM to a tile's full_bbox, writing a per-tile GeoTIFF. Pads
    # edge tiles with nodata where they extend past the DEM extent.
    # @param tile: Tile to crop
    # @param dem_path: source cluster DEM
    # @param output_path: per-tile DEM output
    # @returns: output_path

    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0

        window, dst_transform, dst_width, dst_height = \
            _bbox_to_pixel_aligned_window(
                tile.full_bbox, src.transform, src.width, src.height
            )

        data = _read_window_with_padding(src, window, fill_value=nodata)

        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": dst_transform,
            "nodata": nodata,
            "compress": "LZW",
            "tiled": True,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)

    return output_path


# --------- Canopy Cropping (with optional reproject) ---------

def crop_canopy_to_tile_grid(tile: Tile, canopy_path: Path,
                              dem_tile_path: Path,
                              output_path: Path) -> Path:
    # Crop and reproject canopy to match the per-tile DEM's grid exactly.
    # @param tile: Tile (unused directly; grid comes from dem_tile_path)
    # @param canopy_path: source cluster canopy raster
    # @param dem_tile_path: per-tile DEM defining the target grid
    # @param output_path: per-tile canopy output
    # @returns: output_path

    canopy_path = Path(canopy_path)
    dem_tile_path = Path(dem_tile_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Target grid from the per-tile DEM (already cropped + aligned)
    with rasterio.open(dem_tile_path) as dem_tile:
        dst_transform = dem_tile.transform
        dst_width = dem_tile.width
        dst_height = dem_tile.height
        dst_crs = dem_tile.crs

    # bilinear + nodata caveat: rasterio averages nodata as a numeric value,
    # so a boundary cell next to nodata=255 and a real 60% pixel would get
    # ~157 (garbage). Fix: read to float32, mask nodata to NaN, reproject
    # with NaN nodata so it propagates, then restore the sentinel.
    with rasterio.open(canopy_path) as src:
        src_nodata = src.nodata if src.nodata is not None else 255

        src_data = src.read(1).astype(np.float32)
        src_data[src_data == src_nodata] = np.nan

        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        # Restore the nodata sentinel and cast back to the source dtype
        dst_dtype = src.dtypes[0]
        dst_data[np.isnan(dst_data)] = src_nodata

        if dst_dtype == "uint8":
            dst_data = np.clip(dst_data, 0, 255).round().astype(np.uint8)
        else:
            dst_data = dst_data.astype(dst_dtype)

        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": dst_dtype,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": src_nodata,
            "compress": "LZW",
            "tiled": True,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dst_data, 1)

    return output_path


# --------- Top-Level Wrapper ---------

def crop_tile_inputs(tile: Tile, cluster_id: str,
                      dem_path: Path, canopy_path: Path,
                      work_dir: Path) -> Dict[str, Path]:
    # Crop both DEM and canopy for one tile, caching by output existence.
    # @param tile: Tile to crop
    # @param cluster_id: e.g. "cluster_11"
    # @param dem_path: source cluster DEM
    # @param canopy_path: source cluster canopy
    # @param work_dir: scratch root (writes to work_dir/tiles/<cluster_id>/)
    # @returns: dict with 'dem' and 'canopy' Path keys

    out_dir = _tile_dirname(work_dir, cluster_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_out = out_dir / _tile_filename(tile, "dem")
    canopy_out = out_dir / _tile_filename(tile, "canopy")

    if dem_out.exists() and canopy_out.exists():
        logger.info(f"  tile {tile.tile_id}: cached DEM+canopy in {out_dir}")
        return {"dem": dem_out, "canopy": canopy_out}

    logger.info(f"  tile {tile.tile_id}: cropping DEM and canopy")
    crop_dem_to_tile(tile, dem_path, dem_out)
    crop_canopy_to_tile_grid(tile, canopy_path, dem_out, canopy_out)

    return {"dem": dem_out, "canopy": canopy_out}


# --------- Smoke Test ---------

if __name__ == "__main__":
    import sys
    import argparse
    from .tile_grid import compute_tile_grid

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [tile-crop] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Smoke test for tile cropping")
    parser.add_argument("--cluster-id", default="cluster_11",
                        help="Cluster name (e.g. cluster_11)")
    parser.add_argument("--dem-dir", default="/home/val/regionalRunout/data/DEM")
    parser.add_argument("--canopy-dir", default="/home/val/regionalRunout/data/forests")
    parser.add_argument("--work-dir", default="/home/val/regionalRunout/work")
    parser.add_argument("--core-m", type=float, default=6000.0)
    parser.add_argument("--overlap-m", type=float, default=3000.0)
    args = parser.parse_args()

    cluster_id = args.cluster_id
    dem_path = Path(args.dem_dir) / f"{cluster_id}.tif"
    canopy_path = Path(args.canopy_dir) / f"{cluster_id}.tif"
    work_dir = Path(args.work_dir)

    if not dem_path.exists():
        print(f"ERROR: DEM not found: {dem_path}")
        sys.exit(1)
    if not canopy_path.exists():
        print(f"ERROR: canopy not found: {canopy_path}")
        sys.exit(1)

    print(f"DEM:    {dem_path}")
    print(f"Canopy: {canopy_path}")
    print(f"Work:   {work_dir}")

    with rasterio.open(dem_path) as src:
        dem_bbox = (src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top)
        print(f"DEM bbox:  {dem_bbox}")
        print(f"DEM size:  {src.width} x {src.height}")
        print(f"DEM CRS:   {src.crs}")

    tiles = compute_tile_grid(dem_bbox, core_m=args.core_m, overlap_m=args.overlap_m)
    print(f"\n{len(tiles)} tile(s) for {cluster_id}")
    print()

    # Crop the first 1-2 tiles (fast)
    sample = tiles[:2] if len(tiles) > 2 else tiles
    for t in sample:
        print(f"--- tile {t.tile_id} ---")
        print(f"  full_bbox: {t.full_bbox}")
        paths = crop_tile_inputs(t, cluster_id, dem_path, canopy_path, work_dir)

        with rasterio.open(paths["dem"]) as d:
            valid = (d.read(1) != d.nodata).sum() / (d.width * d.height) * 100
            print(f"  DEM:    {d.width}x{d.height}, {valid:.1f}% valid, "
                  f"crs={d.crs.to_epsg() or 'compound'}")
        with rasterio.open(paths["canopy"]) as c:
            cmean = c.read(1).mean()
            print(f"  CANOPY: {c.width}x{c.height}, mean value={cmean:.1f}, "
                  f"crs={c.crs.to_epsg() or 'compound'}")
        print(f"  -> {paths['dem']}")
        print(f"  -> {paths['canopy']}")

    print("\nSmoke test passed.")
