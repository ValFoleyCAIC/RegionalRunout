"""
Central Configuration Module for Statewide Avalanche Path Mapping

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Centralized parameters for all pipeline components across the three-stage
    workflow (data prep, release area detection, FlowPy runout). The pipeline
    runs in one of three modes, set via GlobalConfig.mode or main.py's --mode:
    "release-only" (stages 1-2, no FlowPy), "flowpy-only" (stages 1 + 3,
    needs --release-areas), and "full" (all three, the default).
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import multiprocessing


# Valid pipeline modes
VALID_MODES = ("full", "release-only", "flowpy-only")


# --------- Path Configuration ---------

@dataclass
class PathConfig:
    # File and directory paths for the pipeline

    # Input data
    dem_dir: Path = Path("/home/val/regionalRunout/data/DEM")
    landfire_evc_path: Path = Path("/home/val/regionalRunout/data/forests/cluster_00.tif")
    # Optional per-cluster canopy lookup directory. When set, the pipeline
    # looks for <canopy_dir>/<dem_basename>.tif per DEM tile, falling back to
    # landfire_evc_path if not found. Empty Path("") = single path for all.
    canopy_dir: Path = Path("/home/val/regionalRunout/data/forests")
    # AOI shapefile, used when clip_to_aoi=True (PRAConfig). Each DEM tile is
    # matched to a feature by index (from the DEM filename via
    # PRAConfig.feature_id_pattern).
    aoi_clip_path: Path = Path("/home/val/regionalRunout/data/aoi/ForecasterClusters.shp")
    # Optional existing release area file (polygon .shp/.gpkg or binary .tif).
    # Empty string = automated detection. Required for --mode flowpy-only.
    release_area_path: str = "/home/val/regionalRunout/data/RA/RA_selected.shp"
    # Output directories
    output_dir: Path = Path("/home/val/regionalRunout/outputs")
    work_dir: Path = Path("/home/val/regionalRunout/work")
    log_dir: Path = Path("/home/val/regionalRunout/logs")

    # DEM file pattern for folder scanning
    dem_pattern: str = "*.tif"

    def __post_init__(self):
        for d in [self.output_dir, self.work_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_release_area_path(self):
        # Return the release area path if it exists, else None.
        # @returns: Path or None
        if not self.release_area_path:
            return None
        p = Path(self.release_area_path)
        return p if p.exists() else None

    def get_canopy_for_dem(self, dem_path):
        # Resolve the canopy raster matching a DEM tile: <canopy_dir>/<dem_stem>.tif
        # if it exists, otherwise landfire_evc_path.
        # @param dem_path: Path to a DEM tile (any extension)
        # @returns: Path to the canopy raster to use
        dem_path = Path(dem_path)
        if self.canopy_dir and Path(self.canopy_dir).exists():
            candidate = Path(self.canopy_dir) / f"{dem_path.stem}.tif"
            if candidate.exists():
                return candidate
        return self.landfire_evc_path


# --------- Performance Configuration ---------

@dataclass
class PerformanceConfig:
    # Parallel processing and resource management.
    #
    # max_workers: number of tile_processor sub-tiles running simultaneously.
    #   Each may itself spawn flowpy_cpu_cap workers during FlowPy, so
    #   effective parallelism is ~max_workers * flowpy_cpu_cap.
    #
    # flowpy_cpu_cap: passed to FlowPy as nCPU for per-release-cell tracing.
    #   Too low causes per-worker memory bloat and OOM kills on tiles with
    #   many release cells; too high competes with max_workers and can also
    #   exhaust RAM.
    #
    # On a 64-core shared server, max_workers=1, flowpy_cpu_cap=16 uses ~25%
    # of CPUs and leaves the rest for other users. .
    max_workers: int = 1
    flowpy_cpu_cap: int = 16
    max_memory_per_worker_gb: float = 15.7
    checkpoint_interval: int = 5

    def __post_init__(self):
        self.max_workers = max(1, self.max_workers)
        self.flowpy_cpu_cap = max(1, self.flowpy_cpu_cap)

        available = multiprocessing.cpu_count()
        max_parallel = available // self.flowpy_cpu_cap
        if self.max_workers > max_parallel:
            self.max_workers = max(1, max_parallel)

    def get_optimal_workers(self):
        # Optimal worker count from CPU and memory limits.
        # @returns: int
        cpu_count = multiprocessing.cpu_count()
        cpu_based = max(1, cpu_count // self.flowpy_cpu_cap)

        try:
            import psutil
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
            mem_based = max(1, int(total_mem_gb / self.max_memory_per_worker_gb))
        except ImportError:
            mem_based = cpu_based

        return max(1, min(cpu_based, mem_based, self.max_workers))


# --------- Vegetation Configuration ---------

@dataclass
class VegetationConfig:
    # Canopy and FSI generation parameters.
    #
    # canopy_source_type selects input handling:
    #   "direct_pct"  : raster values are canopy percent (0-100), e.g. CSFS
    #                   regression output. Reprojected with BILINEAR.
    #   "evc_encoded" : LANDFIRE EVC encoding (110-199 = 10-99% tree cover,
    #                   210-299 = 10-99% shrub cover). NEAREST resampling.

    canopy_source_type: str = "direct_pct"

    # The 4 *_evc fields are used only when canopy_source_type is
    # "evc_encoded" (kept for LANDFIRE inputs).
    tree_cover_min_evc: int = 110
    tree_cover_max_evc: int = 199
    shrub_cover_min_evc: int = 210
    shrub_cover_max_evc: int = 299

    # FSI scaling for the FlowPy forest module: tree_cover_pct / fsi_max_cover,
    # clamped to [0, 1].
    #
    # Ex: 60 means 60% canopy is the maximum forest effect -
    # additional density above 60% doesn't physically add much more
    # flow-stopping capacity (a mature closed-canopy stand is already at full
    # effect). This steepens the gradient where it matters:
    #   30% cover -> FSI 0.50  (moderate forest effect)
    #   45% cover -> FSI 0.75  (substantial forest effect)
    #   60% cover -> FSI 1.00  (max forest effect)
    #
    # If the canopy product is well-calibrated to true crown closure, set
    # this to 100. If it over-predicts density (as many regression products
    # do), keep 60 to compensate at the saturation end.
    fsi_max_cover: float = 60.0

    # Buffer distance (meters) to dilate vegetation mask edges
    veg_buffer_m: float = 0.0

    # Gaussian smoothing radius (meters) for the FSI raster fed to FlowPy.
    # 0 = none; useful for blocky LANDFIRE input (try 15m). For continuous
    # direct-percent products, leave at 0.
    veg_smooth_sigma_m: float = 0.0


# --------- PRA Detection Configuration ---------

@dataclass
class PRAConfig:
    # Potential Release Area detection - Veitinger fuzzy logic with forest
    # density weighting.

    # Cauchy membership: mu(x) = 1 / (1 + ((x - c) / a)^(2b))

    # Slope membership (peak ~38 deg)
    slope_a: float = 10.0
    slope_b: float = 3.0
    slope_c: float = 38.0

    # Forest density membership (logistic sigmoid, monotonic).
    # midpoint: cover % where membership = 0.5
    # steepness: transition width (higher = sharper dropoff)
    #
    # 30% crown cover is the standard threshold above which slab formation is
    # substantially suppressed (avalanche.org guidance, broadly accepted in
    # NA/EU snow mechanics). This is a physical parameter and should NOT be
    # tweaked to compensate for canopy-product bias - calibrate the canopy
    # product instead, or use tree_cover_smooth_sigma_m to average out local
    # over-prediction.
    #
    # At midpoint=30, steepness=10:
    #   15% cover -> mu = 0.82   (open: release likely)
    #   25% cover -> mu = 0.62
    #   30% cover -> mu = 0.50   (midpoint - physical threshold)
    #   35% cover -> mu = 0.38
    #   50% cover -> mu = 0.12   (dense: strongly suppressed)
    #   70% cover -> mu = 0.02
    forest_midpoint: float = 30.0
    forest_steepness: float = 10.0

    # Optional Gaussian smoothing of tree cover % before fuzzy logic. At 1m
    # DEM resolution every tree-shadow pixel can punch a hole through the
    # fuzzy AND ("swiss cheese" polygons); smoothing averages trees into a
    # stand-level density. Sigma in METERS; 0 = none; 10-15m at 1m DEM.
    tree_cover_smooth_sigma_m: float = 10.0

    # Wind shelter (DISABLED by default)
    use_wind_shelter: bool = False
    wind_direction_deg: float = 270.0
    wind_shelter_a: float = 3.0
    wind_shelter_b: float = 2.0
    wind_shelter_c: float = 0.5

    # Fuzzy AND operator gamma (Werners 1988).
    # 0.5 = balanced; higher favors intersection, lower favors union.
    fuzzy_gamma: float = 0.5

    # PRA threshold for binary conversion
    pra_threshold: float = 0.55

    # Patch size filtering (m2)
    min_patch_area_m2: float = 1000.0
    max_patch_area_m2: float = 500000.0

    # Elevation filtering (optional, None = disabled)
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None

    # Sympathetic release buffer
    sympathetic_buffer_m: float = 0.0          # fixed buffer (m), 0 = disabled
    terrain_aware_buffer: bool = False          # expand into adjacent relaxed-slope cells
    terrain_buffer_min_slope_deg: float = 30.0  # relaxed slope threshold for expansion
    terrain_buffer_max_iterations: int = 3      # max expansion iterations

    # Morphological cleanup. With a noisy canopy product, keeping these low
    # avoids over-shrinking real PRAs. Erode 1 then dilate 1 = mild opening
    # that drops single-pixel noise.
    erosion_iterations: int = 1
    dilation_iterations: int = 1

    # AOI clipping: when True, release zones outside the AOI feature are
    # removed AFTER segmentation. FlowPy runout can still extend into the
    # buffer (paths starting near the AOI edge run out normally), but releases
    # can only START inside the AOI feature. Set False for full forecast-zone
    # runs where releases should appear everywhere the canopy is valid.
    clip_to_aoi: bool = True

    # Regex to extract the AOI feature index from DEM filenames. Default
    # matches the first integer, so "cluster_03.tif", "feature_03", "tile_3",
    # "3" all map to feature index 3. On no match the pipeline crashes loud
    # rather than silently producing wrong outputs.
    feature_id_pattern: str = r"(\d+)"


# --------- FlowPy Configuration ---------

@dataclass
class FlowPyConfig:
    # AvaFrame com4FlowPy runout model parameters

    alpha_deg: float = 20.0
    exponent: float = 8.0
    flux_threshold: float = 0.003
    max_z: float = 270.0

    # Internal FlowPy tiling (METERS) - separate from tile_processor's
    # core/overlap.
    #
    # CRITICAL: tile_overlap must be >= max expected path length. If paths
    # run 3km from release zones, an overlap of only 2km truncates paths at
    # FlowPy's internal tile boundaries even when the outer overlap is sized
    # correctly.
    #
    # With 3km max path length the sub-tile FlowPy tiling math doesn't work:
    # tile_size=5km/overlap=3km has zero useful core, tile_size=8km/overlap=3km
    # is 64M cells (large per-worker memory), tile_size=20km/overlap=0 disables
    # internal tiling entirely.
    #
    # Chosen approach: process each 15km tile_processor sub-tile as a single
    # FlowPy internal tile. Path continuity within the sub-tile is guaranteed
    # (no internal boundaries); across sub-tiles it's handled by
    # tile_processor's 3km outer overlap.
    #
    # If per-worker memory becomes a problem (BrokenPipeError / silent OOM),
    # the fix is application-layer release batching in flowpy_runner.py, NOT
    # a smaller tile_size.
    tile_size: int = 20000       # > sub-tile size: single internal pass
    tile_overlap: int = 0        # tile_processor's 3km handles continuity
    cpu_cap: int = 16            # polite on a 64-core server. 39K release cells
                                 # / 16 workers = ~2.5K cells/worker - manageable
                                 # with FlowPy's internal chunk batching.

    # Preview mode - faster rough approximation (skips per-cell path tracking).
    # cellCounts, fpTravelAngle, zDeltaSum, slTravelAngle, routFluxSum,
    # depFluxSum are inaccurate in preview mode.
    preview_mode: bool = False

    # Output file selection (pipe-separated, com4FlowPy naming). Available:
    # zdelta|flux|cellCounts|travelLength|fpTravelAngle|zDeltaSum|
    # slTravelAngle|routFluxSum|depFluxSum
    output_files: str = "flux|zDelta|cellCounts|zDeltaSum|travelLength"

    # Forest module. ForestDetrainment is the default because it adds friction
    # AND removes flux on forested cells - what produces meaningful stoppage in
    # dense canopy. ForestFriction alone only dampens runout;
    # ForestFrictionLayer expects per-cell alpha overrides.
    use_forest: bool = True
    forest_module: str = "ForestDetrainment"  # ForestFriction | ForestDetrainment | ForestFrictionLayer

    # ForestFriction params (used by ForestFriction and ForestDetrainment).
    # Alpha increase on forested cells (degrees), scaled by FSI and falling off
    # as velocity approaches the threshold. Tuned so at FSI=1 dense forest,
    # alpha rises from 20 to ~35 deg; velocity threshold at 20 m/s keeps
    # friction active during the fast-moving part of a dry-snow avalanche.
    max_added_friction: float = 15.0
    min_added_friction: float = 2.0
    vel_threshold_friction: float = 20.0  # m/s

    # ForestDetrainment params (flux removal on forested cells) - the values
    # that actually stop flow through dense forest. max_detrainment is per-cell
    # flux removed at v=0 with FSI=1; scales linearly with FSI and falls off as
    # velocity approaches the threshold. At 0.001, a path crossing ~50 forested
    # cells with FSI=1 loses ~0.05 flux total - enough to terminate paths whose
    # remaining flux is near flux_threshold (0.003).
    max_detrainment: float = 0.001
    min_detrainment: float = 0.00005
    vel_threshold_detrain: float = 20.0  # m/s

    # ForestFrictionLayer param (only used when module = ForestFrictionLayer)
    forest_friction_layer_type: str = "relative"  # absolute | relative

    # Skip forest friction for the first N meters of path (forest inside the
    # release zone itself doesn't dampen the start).
    skip_forest_dist_m: float = 0.0


# --------- DEM Processing Configuration ---------

@dataclass
class DEMProcessingConfig:
    # DEM preprocessing and multi-tile handling

    dem_smooth_sigma: float = 1.5
    slope_algorithm: str = "gradient"

    # Resample to this resolution before processing; None = native. Strongly
    # recommended to resample 1m DEMs to 5-10m for FlowPy (avoids braided
    # flow-routing artifacts from micro-topography) and for PRA detection (so
    # individual trees don't punch holes in the release polygons).
    target_resolution_m: Optional[float] = 5.0

    # Smart edge buffering: buffer pulled from neighboring tiles (meters)
    dem_edge_buffer_m: float = 1000.0


# --------- Master Configuration ---------

class GlobalConfig:
    # Container for all configuration sections

    def __init__(self):
        # Pipeline mode: "full" | "release-only" | "flowpy-only".
        # Overridden via --mode in main.py.
        self.mode: str = "full"

        self.paths = PathConfig()
        self.performance = PerformanceConfig()
        self.vegetation = VegetationConfig()
        self.pra = PRAConfig()
        self.flowpy = FlowPyConfig()
        self.dem = DEMProcessingConfig()

        # Sync FlowPy CPU cap with performance config
        self.flowpy.cpu_cap = self.performance.flowpy_cpu_cap

    def set_mode(self, mode):
        # Set pipeline mode with validation.
        # @param mode: one of VALID_MODES
        # @raises: ValueError if mode is unknown
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of: {VALID_MODES}"
            )
        self.mode = mode

    def run_release_detection(self):
        # @returns: True if this run should compute release areas
        return self.mode in ("full", "release-only")

    def run_flowpy(self):
        # @returns: True if this run should call FlowPy
        return self.mode in ("full", "flowpy-only")

    def summary(self):
        # Human-readable configuration summary.
        # @returns: str
        release_path = self.paths.get_release_area_path()

        lines = [
            "=" * 70,
            "AVALANCHE MAPPING CONFIGURATION",
            "=" * 70,
            "",
            f"  Mode:              {self.mode}",
            f"  DEM directory:     {self.paths.dem_dir}",
            f"  Canopy/EVC path:   {self.paths.landfire_evc_path}",
            f"  Release area file: {release_path if release_path else 'None (auto-detect)'}",
            f"  Output directory:  {self.paths.output_dir}",
            "",
            f"  Workers: {self.performance.max_workers}  |  CPUs per FlowPy: {self.performance.flowpy_cpu_cap}",
            f"  DEM edge buffer:   {self.dem.dem_edge_buffer_m} m",
            f"  DEM resolution:    {str(self.dem.target_resolution_m) + ' m (resampled)' if self.dem.target_resolution_m else 'native'}",
            "",
            f"  PRA threshold:     {self.pra.pra_threshold}",
            f"  Slope peak:        {self.pra.slope_c} deg",
            f"  Forest midpoint:   {self.pra.forest_midpoint}% (steepness={self.pra.forest_steepness})",
            f"  Tree cover smooth: {self.pra.tree_cover_smooth_sigma_m} m sigma",
            f"  Wind shelter:      {'Enabled' if self.pra.use_wind_shelter else 'Disabled'}",
            f"  Min patch area:    {self.pra.min_patch_area_m2} m2",
            "",
            f"  FlowPy alpha:      {self.flowpy.alpha_deg} deg",
            f"  Forest module:     {self.flowpy.forest_module if self.flowpy.use_forest else 'Disabled'}",
            f"  FSI saturation:    {self.vegetation.fsi_max_cover}% canopy",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


def load_config():
    # Load default global configuration.
    # @returns: GlobalConfig instance
    return GlobalConfig()


def validate_config(config):
    # Validate configuration and return warnings.
    # @param config: GlobalConfig instance
    # @returns: list of warning strings (empty if all valid)

    warnings = []

    if config.mode not in VALID_MODES:
        warnings.append(f"Unknown mode {config.mode!r}, expected {VALID_MODES}")

    if config.mode == "flowpy-only":
        if not config.paths.get_release_area_path():
            warnings.append(
                "Mode flowpy-only requires --release-areas to point at a "
                "valid release polygon/raster file"
            )

    if not config.paths.dem_dir.exists():
        warnings.append(f"DEM directory not found: {config.paths.dem_dir}")

    if config.flowpy.use_forest and not config.paths.landfire_evc_path.exists():
        warnings.append(f"Canopy/EVC raster not found: {config.paths.landfire_evc_path}")

    if config.paths.release_area_path and not config.paths.get_release_area_path():
        warnings.append(f"Release area file not found: {config.paths.release_area_path}")

    if not (0 <= config.pra.pra_threshold <= 1):
        warnings.append("PRA threshold must be between 0 and 1")

    if not (20 <= config.pra.slope_c <= 55):
        warnings.append(f"Slope peak {config.pra.slope_c} deg seems unusual (expected 25-50)")

    if not (10 <= config.pra.forest_midpoint <= 70):
        warnings.append(f"Forest midpoint {config.pra.forest_midpoint}% seems unusual (expected 20-60)")

    if config.pra.forest_steepness <= 0:
        warnings.append("Forest steepness must be positive")

    if config.pra.tree_cover_smooth_sigma_m < 0:
        warnings.append("tree_cover_smooth_sigma_m must be >= 0")

    valid_modules = ["ForestFriction", "ForestDetrainment", "ForestFrictionLayer"]
    if config.flowpy.use_forest and config.flowpy.forest_module not in valid_modules:
        warnings.append(f"Unknown forest module: {config.flowpy.forest_module}")

    if config.flowpy.max_detrainment < config.flowpy.min_detrainment:
        warnings.append("max_detrainment must be >= min_detrainment")

    if config.flowpy.max_added_friction < config.flowpy.min_added_friction:
        warnings.append("max_added_friction must be >= min_added_friction")

    if config.vegetation.fsi_max_cover <= 0:
        warnings.append("fsi_max_cover must be positive")

    if config.dem.dem_edge_buffer_m < 0:
        warnings.append("DEM edge buffer must be >= 0")

    if config.dem.target_resolution_m is not None:
        if config.dem.target_resolution_m <= 0:
            warnings.append("Target resolution must be positive")
        elif config.dem.target_resolution_m < 1.0:
            warnings.append(
                f"Target resolution {config.dem.target_resolution_m}m is finer "
                f"than 1m - likely unintended"
            )

    return warnings


if __name__ == "__main__":
    cfg = load_config()
    print(cfg.summary())

    warnings = validate_config(cfg)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nConfiguration valid")
