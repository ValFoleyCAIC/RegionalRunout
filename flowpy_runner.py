"""
FlowPy Runout Modeling Module

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Runs AvaFrame com4FlowPy avalanche runout simulations with forest
    friction and detrainment support. Handles release area input
    (polygon or raster), forest layer generation, and output discovery.
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import configparser
import logging

try:
    from avaframe.com4FlowPy import com4FlowPy as c4
    from avaframe.in3Utils import cfgUtils as _ava_cfg_utils
    FLOWPY_AVAILABLE = True
except ImportError:
    FLOWPY_AVAILABLE = False
    logging.warning("AvaFrame/FlowPy not available - install with: pip install avaframe")

from config import FlowPyConfig

logger = logging.getLogger(__name__)


def _force_avaframe_ncpu(n_cpu):
    # Patch AvaFrame's config loader so nCPU is honored. AvaFrame reads
    # CPU count from avaframeCfg.ini [MAIN]nCPU via getNumberOfProcesses(),
    # NOT from the per-run cfgSetup we pass to com4FlowPyMain - so cpuCount
    # in cfg["GENERAL"] is silently ignored without this.
    # @param n_cpu: CPU count to force
    # @returns: original loader (for optional restore), or None if no FlowPy

    if not FLOWPY_AVAILABLE:
        return None

    original_loader = _ava_cfg_utils.getGeneralConfig

    def _patched_loader(*args, **kwargs):
        cfg = original_loader(*args, **kwargs)
        try:
            cfg["MAIN"]["nCPU"] = str(int(n_cpu))
            cfg["MAIN"]["CPUPercent"] = "100"
        except (KeyError, TypeError):
            pass
        return cfg

    _ava_cfg_utils.getGeneralConfig = _patched_loader
    logger.info(f"  Forced AvaFrame nCPU={n_cpu} via cfgUtils patch")
    return original_loader


# --------- Release Raster Preparation ---------

def prepare_release_raster(release_path, dem_path, output_raster_path):
    # Create binary release raster aligned to DEM grid. Handles polygon
    # (.shp/.gpkg) and raster (.tif) inputs.
    # @param release_path: Path to release area file
    # @param dem_path: Path to DEM for grid alignment
    # @param output_raster_path: Path to write the release raster
    # @returns: Path to release raster

    release_path = Path(release_path)
    output_raster_path = Path(output_raster_path)
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem_shape = (src.height, src.width)
        dem_transform = src.transform
        dem_crs = src.crs

    ext = release_path.suffix.lower()

    if ext in ['.tif', '.tiff']:
        logger.info(f"Using release raster: {release_path.name}")
        with rasterio.open(release_path) as src:
            release_data = (src.read(1) > 0).astype(np.uint8)

    elif ext in ['.shp', '.gpkg', '.geojson']:
        logger.info(f"Rasterizing release polygons: {release_path.name}")
        gdf = gpd.read_file(release_path)
        if gdf.crs != dem_crs:
            gdf = gdf.to_crs(dem_crs)

        shapes_list = [(geom, 1) for geom in gdf.geometry if geom is not None]
        release_data = rasterize(
            shapes=shapes_list,
            out_shape=dem_shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8
        )

    else:
        raise ValueError(f"Unsupported release format: {ext}")

    profile = {
        "driver": "GTiff",
        "height": dem_shape[0],
        "width": dem_shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": dem_crs,
        "transform": dem_transform,
        "nodata": 0,
        "compress": "LZW"
    }
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(release_data, 1)

    n_release = int((release_data > 0).sum())
    logger.info(f"Release raster: {n_release} pixels")

    return output_raster_path


# --------- FlowPy Execution ---------

def run_flowpy(dem_path, release_path, output_dir, flowpy_config,
               fsi_path=None, run_id=None):
    # Run com4FlowPy avalanche runout simulation.
    # @param dem_path: Path to DEM GeoTIFF
    # @param release_path: Path to release areas (.shp/.gpkg/.tif)
    # @param output_dir: Output directory for results
    # @param flowpy_config: FlowPyConfig instance
    # @param fsi_path: Path to FSI raster for forest module (optional)
    # @param run_id: Unique run identifier (auto-generated if None)
    # @returns: dict with output file paths, empty dict on failure

    if not FLOWPY_AVAILABLE:
        raise ImportError("AvaFrame/FlowPy not installed. Run: pip install avaframe")

    if run_id is None:
        run_id = f"flowpy_{datetime.now():%Y%m%d_%H%M%S}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FlowPy expects this directory structure
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = work_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    release_raster_path = work_dir / f"{run_id}_release.tif"
    prepare_release_raster(release_path, dem_path, release_raster_path)

    forest_enabled = flowpy_config.use_forest and fsi_path is not None

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "alpha": str(flowpy_config.alpha_deg),
        "exp": str(flowpy_config.exponent),
        "flux_threshold": str(flowpy_config.flux_threshold),
        "max_z": str(float(max(0.0, min(flowpy_config.max_z, 8848.0)))),

        "infra": "False",
        "forest": str(forest_enabled),
        "forestInteraction": str(forest_enabled),

        "tileSize": str(flowpy_config.tile_size),
        "tileOverlap": str(flowpy_config.tile_overlap),
        "cpuCount": str(flowpy_config.cpu_cap),
        "procPerCPUCore": "1",
        "chunkSize": "50",
        "maxChunks": "500",

        "previewMode": str(flowpy_config.preview_mode),

        "variableUmaxLim": "False",
        "variableAlpha": "False",
        "variableExponent": "False",

        "fluxDistOldVersion": "False",

        "outputFiles": flowpy_config.output_files,
        "outputFileFormat": ".tif",
        "outputNoDataValue": "-9999",

        "uid": run_id,
    }

    # AvaFrame reads ALL forest params when forest=True regardless of active
    # module; all must be present or it throws NoneType errors.
    if forest_enabled:
        # 1.13.2 requires lowercase first char (forestFriction, forestDetrainment)
        forest_module_name = flowpy_config.forest_module[0].lower() + flowpy_config.forest_module[1:]
        cfg["GENERAL"]["forestModule"] = forest_module_name
        cfg["GENERAL"]["forestPath"] = str(Path(fsi_path))

        cfg["GENERAL"]["maxAddedFrictionFor"] = str(flowpy_config.max_added_friction)
        cfg["GENERAL"]["minAddedFrictionFor"] = str(flowpy_config.min_added_friction)
        cfg["GENERAL"]["velThForFriction"] = str(flowpy_config.vel_threshold_friction)

        # Detrainment params actually stop flow
        cfg["GENERAL"]["maxDetrainmentFor"] = str(flowpy_config.max_detrainment)
        cfg["GENERAL"]["minDetrainmentFor"] = str(flowpy_config.min_detrainment)
        cfg["GENERAL"]["velThForDetrain"] = str(flowpy_config.vel_threshold_detrain)

        cfg["GENERAL"]["forestFrictionLayerType"] = flowpy_config.forest_friction_layer_type

        # Skip forest friction for first N meters of path
        cfg["GENERAL"]["skipForestDist"] = str(flowpy_config.skip_forest_dist_m)

    cfg_setup = cfg["GENERAL"]

    cfg_paths = {
        "workDir": work_dir,
        "tempDir": temp_dir,
        "outDir": result_dir,
        "resDir": result_dir,
        "resultDir": result_dir,
        "inDir": work_dir,

        "uid": run_id,
        "runName": run_id,
        "timeString": f"{datetime.now():%Y%m%d_%H%M%S}",

        "demPath": Path(dem_path),
        "releasePath": Path(release_raster_path),

        "outputFiles": flowpy_config.output_files,
        "outputFileFormat": ".tif",
        "outputNoDataValue": "-9999",
        "customDirs": "True",
        "customPaths": "True",
        "deleteTempFolder": "True",
        "deleteTemp": "True",
        "overwriteResults": "True",
    }

    if forest_enabled:
        cfg_paths["forestPath"] = Path(fsi_path)

    logger.info(f"Running FlowPy: {run_id}")
    logger.info(f"  DEM: {dem_path}")
    logger.info(f"  Release: {release_raster_path}")
    logger.info(f"  Alpha: {flowpy_config.alpha_deg} deg")
    if forest_enabled:
        logger.info(f"  Forest: {fsi_path} (module={flowpy_config.forest_module})")
        if flowpy_config.forest_module == "ForestDetrainment":
            logger.info(f"    detrainment: max={flowpy_config.max_detrainment}, "
                        f"min={flowpy_config.min_detrainment}, "
                        f"vel_th={flowpy_config.vel_threshold_detrain} m/s")
    else:
        logger.info("  Forest: Disabled")

    # Force cpu_cap. Without this AvaFrame uses its main config default
    # (~50% of detected cores), causing long runtimes and OOM-driven
    # BrokenPipeErrors when too few workers handle too many release cells.
    _force_avaframe_ncpu(flowpy_config.cpu_cap)

    try:
        c4.com4FlowPyMain(cfg_paths, cfg_setup)
    except Exception as e:
        logger.error(f"FlowPy raised exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"FlowPy failed for {run_id}: {e}") from e

    # AvaFrame can return normally even when workers were OOM-killed mid-run
    # (BrokenPipeError in the mp queue that the parent doesn't propagate).
    # Empty result_dir = silent failure, so verify outputs exist.
    outputs = discover_outputs(result_dir, run_id, flowpy_config)
    found_count = sum(1 for v in outputs.values() if v is not None)
    if found_count == 0:
        msg = (f"FlowPy returned without writing any output files for "
               f"{run_id}. Usually worker OOM kills (BrokenPipeError) or "
               f"another silent failure. Check log for 'BrokenPipeError' or "
               f"'Killed'. Consider reducing cpu_cap, splitting the release "
               f"file, or adding RAM.")
        logger.error(msg)
        raise RuntimeError(msg)

    expected_count = len(flowpy_config.output_files.split("|"))
    if found_count < expected_count:
        logger.warning(f"FlowPy completed with partial outputs: "
                       f"{found_count}/{expected_count} files found. "
                       f"Some outputs may be missing.")

    logger.info(f"FlowPy complete: {run_id}")
    return outputs


# --------- Output Discovery ---------

def discover_outputs(result_dir, run_id, flowpy_config):
    # Discover FlowPy output files after a run. AvaFrame mixes case between
    # config tokens and filenames (config 'zDelta' -> file 'zdelta'), so
    # match case-insensitively on the filename suffix.
    # @param result_dir: Directory containing FlowPy outputs
    # @param run_id: Run identifier
    # @param flowpy_config: FlowPyConfig instance
    # @returns: dict mapping output token -> file path (or None if missing)

    result_dir = Path(result_dir)
    requested = flowpy_config.output_files.split("|")
    all_files = list(result_dir.glob("*.tif"))
    outputs = {}

    for token in requested:
        token_lower = token.lower()
        matches = [f for f in all_files
                   if f.stem.lower().endswith(f"_{token_lower}")]
        outputs[token] = str(matches[0]) if matches else None

    found = sum(1 for v in outputs.values() if v is not None)
    logger.info(f"  Found {found}/{len(requested)} output files")

    return outputs


# --------- Flux Band Classification ---------

def create_flux_bands(flux_path, output_path, p_possible=25.0, p_likely=50.0, p_core=75.0):
    # Create 3-band flux classification (possible/likely/core).
    # @param flux_path: Path to flux GeoTIFF from FlowPy
    # @param output_path: Output path for classified raster
    # @param p_possible: Percentile threshold for "possible" zone
    # @param p_likely: Percentile threshold for "likely" zone
    # @param p_core: Percentile threshold for "core" zone
    # @returns: Path to flux bands raster, or None on failure

    try:
        with rasterio.open(flux_path) as src:
            flux = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata if src.nodata is not None else -9999.0

        valid = np.isfinite(flux) & (flux != nodata) & (flux > 0)

        if valid.sum() == 0:
            logger.warning("No positive flux values for banding")
            return None

        flux_pos = flux[valid]
        thresh_possible = np.percentile(flux_pos, p_possible)
        thresh_likely = np.percentile(flux_pos, p_likely)
        thresh_core = np.percentile(flux_pos, p_core)

        bands = np.zeros_like(flux, dtype=np.uint8)
        bands[valid & (flux >= thresh_possible)] = 1
        bands[valid & (flux >= thresh_likely)] = 2
        bands[valid & (flux >= thresh_core)] = 3

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": bands.shape[0],
            "width": bands.shape[1],
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "nodata": 0,
            "compress": "LZW"
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(bands, 1)
            dst.write_colormap(1, {
                0: (0, 0, 0, 0),
                1: (255, 255, 80, 255),   # possible - yellow
                2: (255, 170, 40, 255),   # likely - orange
                3: (200, 60, 20, 255)     # core - red
            })

        logger.info(f"Created flux bands: {output_path.name}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to create flux bands: {e}")
        return None
