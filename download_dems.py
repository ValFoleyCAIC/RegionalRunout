"""
DEM Downloader for regionalRunout

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Downloads USGS 3DEP 1m DEMs for each feature in an AOI shapefile. Each
    feature's bounding box is buffered before download so FlowPy runout
    isn't cut off at AOI edges. Two USGS datasets are tried in order via the
    TNMAccess API - the seamless 1m DEM (better quality, newer, may have
    coverage gaps) then the project-based 1m DEM (wider coverage, possible
    small tile-boundary mismatches). The API returns 10x10 km tiles
    intersecting each bbox; these are downloaded, mosaicked, clipped to the
    buffered feature bbox, and written as one .tif per feature to <out_dir>.
    Filenames encode the feature ID so main.py's spatial index treats each
    feature as a single tile.
"""

import argparse
import re
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping

logger = logging.getLogger(__name__)


# --------- TNMAccess API ---------

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Dataset name strings as TNMAccess expects them. Order matters: seamless
# first (better quality), fall back to project-based.
DATASET_PREFERENCE = [
    "Seamless 1-meter Digital Elevation Model (DEM)",
    "Digital Elevation Model (DEM) 1 meter",
]


def query_tnm(bbox_wgs84, dataset_name, max_results=200):
    # Query TNMAccess for products intersecting bbox_wgs84.
    # @param bbox_wgs84: (minx, miny, maxx, maxy) in lon/lat (EPSG:4326)
    # @param dataset_name: exact USGS dataset string from DATASET_PREFERENCE
    # @param max_results: page size (TNM caps at 1000; 200 is plenty per feature)
    # @returns: list of product dicts (urls, sizes, etc.)

    minx, miny, maxx, maxy = bbox_wgs84
    params = {
        "datasets": dataset_name,
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "prodFormats": "GeoTIFF",
        "max": str(max_results),
        "outputFormat": "JSON",
    }
    url = f"{TNM_API}?{urlencode(params)}"

    req = Request(url, headers={"User-Agent": "regionalRunout/1.0"})

    # Polite pre-query delay: TNM silently rate-limits with empty results
    # when hammered. 3s between queries is conservative.
    time.sleep(3.0)

    # Retry on network errors, JSON parse errors (TNM serves HTML error pages
    # as 200s), and empty results (silent rate-limit).
    for attempt in range(4):
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                # HTML error page with status 200 - retryable
                snippet = raw[:80].decode("utf-8", errors="replace") if raw else ""
                logger.warning(f"  TNM returned non-JSON response "
                               f"(attempt {attempt+1}): {snippet!r}")
                if attempt < 3:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise

            items = data.get("items", [])
            total = data.get("total", 0)
            if items:
                return items
            # Zero items: real coverage gap or silent rate-limit
            if attempt < 3:
                wait = 10 * (attempt + 1)
                logger.warning(f"  TNM returned 0 items (total={total}); "
                               f"retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            return items
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"  TNM query attempt {attempt+1} failed: {e}")
            if attempt < 3:
                time.sleep(15 * (attempt + 1))
            else:
                raise


_GRID_CELL_RE = re.compile(r"_x(\d+)y(\d+)_", re.IGNORECASE)


def _grid_cell_id(item, url):
    # Parse the x{N}y{M} grid cell ID from a TNM item's filename. Tries the
    # URL first (always underscore form _x34y432_), then the title.
    # @param item: TNMAccess product dict
    # @param url: download URL
    # @returns: (int, int) tuple or None
    for candidate in (url, item.get("title") or ""):
        m = _GRID_CELL_RE.search(candidate)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


def pick_best_products(items):
    # Deduplicate items covering the same 10x10 km grid cell (different lidar
    # projects sometimes both publish a tile for the same x{N}y{M} cell).
    # Within a cell, prefer larger file size (more data, fewer voids in
    # mountainous terrain), breaking ties with newer publicationDate. Items
    # from different cells are always kept.
    # @param items: list of TNMAccess product dicts
    # @returns: list of (item, download_url) tuples

    # Group by grid cell; items without a parseable cell ID get a unique
    # fallback key so they're never dedup'd against each other.
    by_cell = {}
    for idx, it in enumerate(items):
        url = it.get("downloadURL") or it.get("urls", {}).get("TIFF")
        if not url:
            continue
        cell = _grid_cell_id(it, url)
        if cell is None:
            cell = ("__fallback__", idx)

        size = it.get("sizeInBytes", 0) or 0
        date = it.get("publicationDate") or it.get("dateCreated") or ""

        existing = by_cell.get(cell)
        if existing is None:
            by_cell[cell] = (it, url, size, date)
        else:
            _, _, ex_size, ex_date = existing
            # Prefer larger file; break tie with newer date
            if (size > ex_size) or (size == ex_size and date > ex_date):
                by_cell[cell] = (it, url, size, date)

    return [(it, url) for (it, url, _, _) in by_cell.values()]


# --------- Download ---------

def download_file(url, out_path, chunk_size=1024 * 1024):
    # Stream a file to disk, resuming partial downloads via a Range header.
    # @param url: full HTTP(S) URL
    # @param out_path: target path (Path)
    # @param chunk_size: read buffer size
    # @returns: out_path on success, None on failure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from a leftover .tmp if present, else start fresh
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    start_byte = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {"User-Agent": "regionalRunout/1.0"}
    if start_byte:
        headers["Range"] = f"bytes={start_byte}-"
        logger.info(f"  Resuming from byte {start_byte:,}")

    req = Request(url, headers=headers)
    mode = "ab" if start_byte else "wb"

    try:
        with urlopen(req, timeout=120) as resp, open(tmp_path, mode) as f:
            total = resp.headers.get("Content-Length")
            total = int(total) + start_byte if total else None
            downloaded = start_byte
            last_log = time.time()

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                # Log progress every 5s for big files
                if time.time() - last_log > 5 and total:
                    pct = 100 * downloaded / total
                    logger.info(f"  {pct:5.1f}%  {downloaded/(1024**2):.0f} MB / "
                                f"{total/(1024**2):.0f} MB")
                    last_log = time.time()

        tmp_path.rename(out_path)
        return out_path
    except (HTTPError, URLError, TimeoutError) as e:
        logger.error(f"  Download failed: {e}")
        # Keep .tmp for resume on next run
        return None


# --------- AOI -> Buffered Bbox ---------

def feature_to_buffered_bbox(geom, source_crs, buffer_m, target_crs="EPSG:4326"):
    # Buffer a feature by buffer_m meters and return (bbox_native, bbox_wgs84).
    # bbox_native (source_crs) is used for final clipping; bbox_wgs84 for the
    # TNM query. Buffering is done in a meter-based UTM projection picked from
    # the feature centroid.
    # @param geom: feature geometry
    # @param source_crs: AOI CRS
    # @param buffer_m: buffer distance (m)
    # @param target_crs: query CRS (default EPSG:4326)
    # @returns: (bounds_native_tuple, bounds_wgs84_tuple)

    src = gpd.GeoSeries([geom], crs=source_crs)

    centroid_wgs = src.to_crs("EPSG:4326").geometry.centroid.iloc[0]
    utm_zone = int((centroid_wgs.x + 180) // 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"  # northern hemisphere

    buffered = src.to_crs(utm_crs).buffer(buffer_m)

    native = buffered.to_crs(source_crs).total_bounds  # (minx, miny, maxx, maxy)
    wgs = buffered.to_crs(target_crs).total_bounds

    return tuple(native), tuple(wgs)


# --------- Per-Feature Workflow ---------

def process_feature(feature_idx, feature, source_crs, out_dir, buffer_m,
                    download_cache_dir, dry_run=False, feature_id_attr=None):
    # Download, mosaic, and clip a buffered DEM for one AOI feature.
    # @param feature_idx: integer index for filename fallback
    # @param feature: a GeoDataFrame row
    # @param source_crs: AOI shapefile CRS
    # @param out_dir: where to write the final per-feature .tif
    # @param buffer_m: buffer distance in meters
    # @param download_cache_dir: where raw TNM tiles are cached
    # @param dry_run: if True, only print what would be done
    # @param feature_id_attr: optional attribute name to use in output filename
    # @returns: Path to output .tif, or None if failed/skipped

    if feature_id_attr and feature_id_attr in feature and feature[feature_id_attr]:
        raw_id = str(feature[feature_id_attr])
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_id)
        feature_name = f"cluster_{feature_idx:02d}"
    else:
        feature_name = f"cluster_{feature_idx:02d}"

    out_path = Path(out_dir) / f"{feature_name}.tif"

    if out_path.exists() and not dry_run:
        logger.info(f"[{feature_name}] already exists, skipping")
        return out_path

    logger.info(f"\n[{feature_name}] processing")

    geom = feature.geometry
    if geom is None or geom.is_empty:
        logger.warning(f"[{feature_name}] empty geometry, skipping")
        return None

    bbox_native, bbox_wgs = feature_to_buffered_bbox(
        geom, source_crs, buffer_m
    )
    logger.info(f"  Buffered bbox (native): {tuple(round(v, 1) for v in bbox_native)}")
    logger.info(f"  Buffered bbox (wgs84): {tuple(round(v, 4) for v in bbox_wgs)}")

    # Query TNM, preferring seamless
    items = []
    chosen_dataset = None
    for ds in DATASET_PREFERENCE:
        logger.info(f"  Querying TNM: {ds!r}")
        items = query_tnm(bbox_wgs, ds)
        if items:
            chosen_dataset = ds
            logger.info(f"  Found {len(items)} tile(s) in '{ds}'")
            break
        else:
            logger.info(f"  No tiles in '{ds}'")

    if not items:
        logger.error(f"[{feature_name}] NO 1m DEM coverage found in any dataset")
        return None

    products = pick_best_products(items)
    logger.info(f"  After dedup: {len(products)} unique tile(s)")

    if dry_run:
        for item, url in products:
            size_mb = item.get("sizeInBytes", 0) / (1024**2)
            logger.info(f"    -> {Path(url).name}  ({size_mb:.1f} MB)")
        return None

    # Download every tile
    cache_dir = Path(download_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []

    for item, url in products:
        fname = Path(url.split("?")[0]).name  # strip query string
        local = cache_dir / fname
        if local.exists():
            logger.info(f"  Cached: {fname}")
        else:
            logger.info(f"  Downloading: {fname}")
            result = download_file(url, local)
            if result is None:
                logger.error(f"  Failed to download {fname}, will retry on next run")
                return None
        local_paths.append(local)

    # Mosaic + clip to buffered feature bounds
    logger.info(f"  Mosaicking {len(local_paths)} tile(s) and clipping...")
    success = mosaic_and_clip(local_paths, bbox_native, source_crs, out_path)
    if success:
        logger.info(f"[{feature_name}] -> {out_path}")
        return out_path
    else:
        logger.error(f"[{feature_name}] mosaic/clip failed")
        return None


def mosaic_and_clip(tif_paths, bbox_target, target_crs, out_path):
    # Merge GeoTIFFs and clip to bbox_target (in target_crs). Handles three
    # cases: all sources match target CRS (simple merge); all share one
    # non-target CRS (merge then warp); mixed source CRSes at UTM zone
    # boundaries (warp each outlier to target CRS first, then merge).
    # @param tif_paths: source GeoTIFF paths
    # @param bbox_target: clip bbox in target_crs
    # @param target_crs: output CRS
    # @param out_path: output .tif
    # @returns: True on success, False on any error

    src_files = []
    try:
        for p in tif_paths:
            src_files.append(rasterio.open(p))

        src_crses = [str(s.crs) for s in src_files]
        unique_crses = set(src_crses)
        target_crs_str = str(target_crs)

        if unique_crses == {target_crs_str}:
            # Case 1: all match target CRS
            return _simple_merge_clip(src_files, bbox_target, target_crs, out_path)

        if len(unique_crses) == 1:
            # Case 2: all sources share one non-target CRS
            logger.info(f"  Reprojecting from {src_files[0].crs} to {target_crs} during clip")
            return _mosaic_clip_with_warp(src_files, bbox_target, target_crs, out_path)

        # Case 3: mixed CRSes (e.g. UTM zone 12 + 13). Reproject each
        # non-target tile to target CRS, write a temp file, reopen, merge.
        # Slow but correct.
        logger.info(f"  Mixed source CRSes detected ({len(unique_crses)} CRSes): "
                    f"pre-warping outliers to {target_crs}")

        from rasterio.warp import calculate_default_transform, reproject, Resampling

        warped_paths = []
        for src in src_files:
            if str(src.crs) == target_crs_str:
                warped_paths.append(Path(src.name))
                continue

            warped = Path(src.name).with_name(Path(src.name).stem + "_warp_to_target.tif")
            if warped.exists():
                warped_paths.append(warped)
                continue

            logger.info(f"    pre-warp {Path(src.name).name}: {src.crs} -> {target_crs}")
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height,
                *src.bounds, resolution=1.0
            )
            profile = src.profile.copy()
            profile.update({
                "crs": target_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": -9999.0,
                "compress": "LZW",
                "tiled": True,
            })
            with rasterio.open(warped, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=-9999.0,
                )
            warped_paths.append(warped)

        # Close originals; reopen warped versions for merge
        for s in src_files:
            s.close()
        src_files = []

        warped_open = [rasterio.open(p) for p in warped_paths]
        try:
            return _simple_merge_clip(warped_open, bbox_target, target_crs, out_path)
        finally:
            for s in warped_open:
                s.close()

    except Exception as e:
        logger.error(f"  mosaic_and_clip failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    finally:
        for s in src_files:
            try:
                s.close()
            except Exception:
                pass


def _simple_merge_clip(src_files, bbox_target, target_crs, out_path):
    # Same-CRS merge path. Helper for mosaic_and_clip.
    mosaic, mosaic_transform = merge(
        src_files,
        bounds=bbox_target,
        nodata=-9999.0,
    )

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": mosaic_transform,
        "compress": "LZW",
        "tiled": True,
        "nodata": -9999.0,
    })

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    return True


def _mosaic_clip_with_warp(src_files, bbox_target, target_crs, out_path):
    # Mosaic + reproject to target_crs + clip to bbox_target. Used when TNM
    # tiles arrive in a different CRS than the AOI shapefile: merge in source
    # CRS, warp the whole mosaic to target_crs in-memory, then clip.

    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.io import MemoryFile

    mosaic, mosaic_transform = merge(src_files, nodata=-9999.0)
    src_crs = src_files[0].crs

    src_height, src_width = mosaic.shape[1], mosaic.shape[2]
    src_bounds = rasterio.transform.array_bounds(src_height, src_width, mosaic_transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, src_width, src_height,
        *src_bounds, resolution=1.0
    )

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=mosaic.dtype,
            crs=target_crs,
            transform=dst_transform,
            nodata=-9999.0,
        ) as warp_ds:
            reproject(
                source=mosaic,
                destination=rasterio.band(warp_ds, 1),
                src_transform=mosaic_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=-9999.0,
                dst_nodata=-9999.0,
            )

        with memfile.open() as warped:
            clip_geom = box(*bbox_target)
            clipped, clip_transform = rasterio_mask(
                warped, [mapping(clip_geom)], crop=True, nodata=-9999.0
            )
            out_meta = warped.meta.copy()
            out_meta.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clip_transform,
                "compress": "LZW",
                "tiled": True,
                "nodata": -9999.0,
            })
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(clipped)

    return True


# --------- CLI ---------

def main():
    p = argparse.ArgumentParser(
        description="Download buffered 1m DEMs from USGS TNM for each AOI feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--aoi", type=Path, required=True,
                   help="Path to AOI shapefile/geopackage (any vector format)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for per-feature DEMs (typically data/DEM)")
    p.add_argument("--buffer-m", type=float, default=2000.0,
                   help="Buffer distance around each feature bbox, meters (default 2000)")
    p.add_argument("--feature-index", type=int, default=None,
                   help="Process only this feature index (for smoke testing)")
    p.add_argument("--feature-id-attr", type=str, default=None,
                   help="Shapefile attribute to embed in output filename (e.g. 'name')")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where to cache raw TNM tile downloads "
                        "(default: <out>/_raw_tnm_cache)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded without downloading")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [dem-download] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.aoi.exists():
        logger.error(f"AOI file not found: {args.aoi}")
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out.parent / "_raw_tnm_cache")

    logger.info(f"AOI:         {args.aoi}")
    logger.info(f"Output dir:  {args.out}")
    logger.info(f"Cache dir:   {cache_dir}")
    logger.info(f"Buffer:      {args.buffer_m} m")

    aoi = gpd.read_file(args.aoi)
    logger.info(f"Loaded {len(aoi)} feature(s) in CRS {aoi.crs}")

    if args.feature_index is not None:
        if args.feature_index >= len(aoi):
            logger.error(f"--feature-index {args.feature_index} out of range")
            sys.exit(1)
        indices = [args.feature_index]
    else:
        indices = list(range(len(aoi)))

    successes = 0
    failures = []

    for idx in indices:
        try:
            result = process_feature(
                idx, aoi.iloc[idx], aoi.crs,
                args.out, args.buffer_m, cache_dir,
                dry_run=args.dry_run,
                feature_id_attr=args.feature_id_attr,
            )
            if result is not None:
                successes += 1
            else:
                if not args.dry_run:
                    failures.append(idx)
        except Exception as e:
            logger.exception(f"[feature {idx}] unhandled error: {e}")
            failures.append(idx)

    logger.info("\n" + "=" * 60)
    logger.info(f"Done. {successes}/{len(indices)} features processed")
    if failures:
        logger.warning(f"Failed feature indices: {failures}")
        logger.warning("Re-run the command — failed downloads will resume from cache")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
