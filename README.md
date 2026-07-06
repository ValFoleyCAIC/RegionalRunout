# regionalRunout

Statewide avalanche release-area detection and runout modeling pipeline
for Colorado. Turns 1 m LiDAR DEMs and a canopy-cover product into a
two-class polygon layer, release zones and modeled runout zones, 
organized within CAIC forecast zones, for public-facing backcountry hazard
communication and recreation planning.

Author: Valerie Foley
Last Updated: 7/2026


## Pipeline overview

The pipeline runs statewide by tiling each forecast-zone "cluster" into
overlapping sub-tiles (9 km core, 3 km overlap → 15 km processing extent),
processing each sub-tile through all stages, then mosaicking per cluster.

### Stages

| Stage | Module | Output |
|-------|--------|--------|
| Terrain analysis | `terrain_analysis.py` | slope, aspect, curvature, (optional) wind shelter |
| Vegetation / FSI | `vegetation_mask.py` | tree-cover % (for PRA) and Forest Structure Index (for FlowPy) |
| Release detection | `release_detection.py` | fuzzy PRA raster → individualized release polygons |
| Runout modeling | `flowpy_runner.py` | com4FlowPy flux / zDelta / cellCounts / travelLength |
| Path extraction | `path_extraction.py` | per-path polygons and single runout-boundary polygon |
| Orchestration | `main.py` | runs the three stages per tile, writes summaries |

### Tiled statewide runner (`tile_processor/`)

| Module | Role |
|--------|------|
| `tile_grid.py` | compute the core/overlap tile grid for a cluster |
| `tile_input_cropping.py` | crop DEM + canopy to each sub-tile |
| `tile_runner.py` | run `main.process_single_tile` on one cropped sub-tile |
| `raster_mosaicker.py` | max-merge per-tile outputs into cluster rasters |


## Repository layout

```
regionalRunout/
├── README.md
├── environment.yml
├── .gitignore
│
├── config.py
├── main.py
├── terrain_analysis.py
├── vegetation_mask.py
├── release_detection.py
├── flowpy_runner.py
├── path_extraction.py
├── download_dems.py
├── resource_monitor.py
├── adaptive_throttle.py
├── schedule_runner.py
│
├── tile_processor/
│   ├── __init__.py
│   ├── tile_grid.py
│   ├── tile_input_cropping.py
│   ├── tile_runner.py
│   └── raster_mosaicker.py
│
└── scripts/
    ├── run_all_clusters_scheduled.sh
    ├── run_all_clusters_release_only.sh
    ├── run_selected_releases.sh
    ├── flowpy_progress.sh
    └── benchmark_server.sh
```


## Setup

### 1. Create the environment

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate avalanche_mapping
```

`scikit-image` (watershed segmentation) and `avaframe>=1.13.2` (com4FlowPy)
are required; the geo stack (rasterio/GDAL/PROJ/GEOS/shapely/geopandas) is
pinned via conda-forge to avoid binary mismatches.

### 2. Check install

```bash
python -c "import rasterio, geopandas, numpy, scipy, skimage; \
           from avaframe.com4FlowPy import com4FlowPy; print('OK')"
```

### 3. Edit `config.py`

Edit the paths in `config.PathConfig`, or override on the `main.py` command
line with `--dem-dir`, `--evc-path`, `--canopy-dir`, `--aoi-clip-path`,
`--release-areas`, `--output-dir`.


## Running

### Modes

The pipeline has three modes (`--mode`, default `full`):

| Mode | Stages | Requirements |
|------|--------|-------|
| `full` | terrain + release detection + FlowPy | — |
| `release-only` | terrain + release detection | — |
| `flowpy-only` | terrain + FlowPy on existing releases | `--release-areas` |

### Single-DEM check

```bash
python main.py --verbose --monitor --throttle --low-priority \
    --dem-dir   /path/to/one_dem_folder \
    --output-dir /path/to/outputs/ \
    --evc-path  /path/to/canopy.tif \
    --workers 1
```

Use a folder containing a single small DEM. Watch `logs/pipeline.log`.

### One cluster sub-tile (tiled runner)

```bash
python -m tile_processor.tile_runner \
    --cluster-id cluster_11 --tile-id 00_00 \
    --output-dir /path/to/outputs
```

### All clusters, scheduled (production)

`scripts/run_all_clusters_scheduled.sh` runs clusters in size order inside a
time window (11:00–04:00 MDT), blocking 04:00–11:00, auto-mosaicking each
cluster when its tiles finish. For a supervised background run with graceful
day/night cutoff:

```bash
nohup python schedule_runner.py \
    --output-dir   /path/to/outputs \
    --window-start 12:00 --window-end 04:00 --tz America/Denver \
    --worker-cmd "python main.py --monitor --throttle --low-priority \
                  --dem-dir /path/to/data/DEM \
                  --output-dir /path/to/outputs \
                  --evc-path /path/to/canopy.tif" \
    > /path/to/logs/supervisor.log 2>&1 &
disown
```

The supervisor writes `STOP_AFTER_TILE` when the window closes; `main.py`
checks it between tiles, finishes the current tile, and exits cleanly.
Progress: `scripts/flowpy_progress.sh`. Server sizing before a run:
`scripts/benchmark_server.sh` (add `--watch 30` to catch load swings).

### Download DEMs

```bash
python download_dems.py --aoi /path/to/aoi.shp --out /path/to/data/DEM --buffer-m 2000
```

Downloads buffered 1 m DEMs per AOI feature from USGS TNM, mosaics, clips, and
writes one `cluster_XX.tif` per feature.


## Key parameters (see `config.py` for the rationale on each)

| Parameter | Default | Notes |
|-----------|---------|-------|
| PRA threshold | 0.55 | elevated vs. published 0.30 to offset canopy over-prediction |
| Forest membership midpoint | 30 % cover | physical slab-suppression threshold |
| FSI saturation (`fsi_max_cover`) | 60 % | canopy % at which forest effect maxes out |
| FlowPy alpha / exponent | 20° / 8 | com4FlowPy runout controls |
| FlowPy flux threshold | 0.003 | runout limit |
| Forest module | `ForestDetrainment` | adds friction and removes flux in canopy |
| Tile core / overlap | 9 km / 3 km | outer tiling; FlowPy internal tiling disabled |
| Processing resolution | 5 m | validated; resampled from 1 m |
| CPU cap | 16 | ~25 % of a 64-core shared server |



## Status/Limitations

### Hardcoded assumptions to parameterize

This was built to run on one shared 64-core server, so a few things are hardcoded 
and would need changing to run elsewhere:

- **Absolute paths:** `config.PathConfig`, `download_dems.py`, and the `scripts/`
  wrappers all point at `/home/val/regionalRunout/...`. Can override these in
  the `main.py` command line, or swap the `PathConfig` defaults to read from an environment
  variable.
- **Code location in the scripts:** The `scripts/*.sh` wrappers `cd` into a `pipeline/`
  directory and `sys.path.insert` that same path in their embedded Python, so those
  two lines need updating if the code lives somewhere else.
- **CPU/RAM tuning:** `flowpy_cpu_cap=16` and `max_workers=1` are set to use ~25% of
  the CAIC server so it stays 'polite' on a shared machine. On a dedicated server these
  are too conservative.
- **Day/night window:** `schedule_runner.py` and `run_all_clusters_scheduled.sh` pause
  from 04:00–11:00 MDT to stay out of other users' way. On a dedicated server we don't
  need this.

### Not yet built

- **Parallel processing:** Everything runs sequentially right now (`max_workers=1`). That's
  fine for the current hardware, but may need to be added depending on the final setup. Note:
  across tiles/clusters processing is sequential, for flowpy within a tile it runs parallel.
- **Skip-resume in the tiled runner:** `tile_runner.is_tile_done()` looks for
  `pipeline_summary.json`, but `process_single_tile` actually writes `{tile_name}_summary.json`,
  so the skip doesn't currently run. The shell runners get around this by grepping logs for
  `status=complete`. This isn't a huge fix if we do spot and need stop/resume.
- **Polygon merger, memory estimator, and parallel scheduler** aren't built yet.


## References

- Veitinger, Purves & Sovilla (2016) — fuzzy-logic potential release area detection
- Bühler et al. (2018) — aspect-weighted release segmentation
- Duvillier et al. (2023) — watershed-based release delineation
- Bühler et al. (2022) — statewide avalanche polygon mapping
- D'Amboise et al. (2022) — Flow-Py runout model
- Teich et al. (2019) — forest friction / detrainment
- Toft et al. (2024) — AutoATES v2.0
- Greene et al. (2022) — SWAG / Relative Size (R4)
- Colorado Geological Survey — avalanche fatalities
- AvaFrame com4FlowPy documentation
