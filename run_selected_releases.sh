#!/bin/bash
# Run FlowPy on selected release areas across multiple clusters.
#
# Takes a single shapefile/gpkg of hand-picked release polygons (possibly
# spanning many clusters), spatially clips it per cluster, and runs the
# per-tile pipeline in flowpy-only mode against the cluster-local release
# subset.
#
# Tiles whose core doesn't intersect any selected release are skipped
# automatically - no wasted FlowPy runs on empty tiles.
#
# Usage:
#   ./run_selected_releases.sh <selected_releases.shp> [output_subdir]
#
# Example:
#   ./run_selected_releases.sh /home/val/picks/my_selected_releases.shp \
#                              outputs_flowpy_selected

set -u
cd /home/val/regionalRunout/pipeline

if [ $# -lt 1 ]; then
    echo "Usage: $0 <selected_releases.shp_or_gpkg> [output_subdir]"
    exit 1
fi

RELEASES_FULL="$1"
OUT_SUBDIR="${2:-outputs_flowpy_selected}"

if [ ! -f "$RELEASES_FULL" ]; then
    echo "ERROR: release file not found: $RELEASES_FULL"
    exit 1
fi

LOGDIR=/home/val/regionalRunout/logs
DEMDIR=/home/val/regionalRunout/data/DEM
WORKDIR=/home/val/regionalRunout/work/selected_releases
OUTDIR=/home/val/regionalRunout/${OUT_SUBDIR}

mkdir -p "$WORKDIR" "$OUTDIR" "$LOGDIR"

CLUSTERS="${CLUSTERS:-11 06 07 02 14 03 05 15 08 01 17 00}"

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Flowpy-only with selected releases"
echo "  Input:  $RELEASES_FULL"
echo "  Output: $OUTDIR"
echo "============================================================"

for cluster_num in $CLUSTERS; do
    CLUSTER="cluster_${cluster_num}"
    DEM_PATH="${DEMDIR}/${CLUSTER}.tif"

    if [ ! -f "$DEM_PATH" ]; then
        continue
    fi

    # Spatially clip the selected-releases file to this cluster's DEM
    # bbox. Writes a per-cluster gpkg if anything falls inside.
    CLUSTER_RELEASES="$WORKDIR/${CLUSTER}_selected.gpkg"
    FILTER_LOG="$LOGDIR/${CLUSTER}_release_filter.log"

    python > "$FILTER_LOG" 2>&1 <<PYEOF
import rasterio
import geopandas as gpd
from shapely.geometry import box

with rasterio.open("$DEM_PATH") as src:
    bbox = box(src.bounds.left, src.bounds.bottom,
               src.bounds.right, src.bounds.top)
    dem_crs = src.crs

gdf = gpd.read_file("$RELEASES_FULL")
if gdf.crs != dem_crs:
    gdf = gdf.to_crs(dem_crs)

hit = gdf[gdf.intersects(bbox)].copy()
if len(hit) == 0:
    print("NO_RELEASES")
else:
    hit.to_file("$CLUSTER_RELEASES", driver="GPKG", layer="releases")
    print(f"OK {len(hit)}")
PYEOF

    RESULT=$(tail -1 "$FILTER_LOG")
    if [[ "$RESULT" == "NO_RELEASES" ]]; then
        echo "$(date '+%H:%M:%S') $CLUSTER: no selected releases - skipping"
        continue
    fi
    if [[ ! "$RESULT" =~ ^OK ]]; then
        echo "$(date '+%H:%M:%S') $CLUSTER: filter failed (see $FILTER_LOG)"
        continue
    fi
    N_REL=$(echo "$RESULT" | awk '{print $2}')

    echo
    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Starting $CLUSTER ($N_REL releases)"
    echo "============================================================"

    # Compute tile grid, then keep only tiles whose core intersects a
    # selected release. Skip-list saves real time on big sparse clusters.
    TILE_IDS=$(python <<PYEOF
import rasterio, sys
import geopandas as gpd
from shapely.geometry import box
sys.path.insert(0, '/home/val/regionalRunout/pipeline')
from tile_processor.tile_grid import compute_tile_grid

with rasterio.open("$DEM_PATH") as src:
    bbox = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

gdf = gpd.read_file("$CLUSTER_RELEASES")
release_union = gdf.unary_union

tiles = compute_tile_grid(bbox, core_m=9000.0, overlap_m=3000.0)
for t in tiles:
    if box(*t.core_bbox).intersects(release_union):
        print(t.tile_id)
PYEOF
)
    if [ -z "$TILE_IDS" ]; then
        echo "$(date '+%H:%M:%S') $CLUSTER: tile filtering returned no tiles - skipping"
        continue
    fi

    N_TILES=$(echo "$TILE_IDS" | wc -l)
    echo "  $N_TILES tiles intersect selected releases"

    TILE_NUM=0
    for tile in $TILE_IDS; do
        TILE_NUM=$((TILE_NUM + 1))

        LOG="$LOGDIR/${CLUSTER}_tile_${tile}_flowpy.log"
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $TILE_NUM/$N_TILES ($tile) starting"

        python -m tile_processor.tile_runner \
            --cluster-id "$CLUSTER" \
            --tile-id "$tile" \
            --mode flowpy-only \
            --release-areas "$CLUSTER_RELEASES" \
            --output-dir "$OUTDIR" \
            > "$LOG" 2>&1

        if grep -q "status=complete" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: complete"
        elif grep -q "status=skipped" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: skipped (already done)"
        elif grep -q "status=failed\|RuntimeError.*FlowPy\|BrokenPipeError" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: FAILED (see $LOG)"
        else
            echo "$(date '+%H:%M:%S')   tile $tile: status unclear (see $LOG)"
        fi
    done
done

echo
echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') ALL CLUSTERS COMPLETE"
echo "============================================================"
echo "FlowPy outputs:        $OUTDIR/<cluster>/tiles/tile_<row>_<col>/flowpy/"
echo "Runout boundary polys: $OUTDIR/<cluster>/tiles/tile_<row>_<col>/paths/*_runout_boundary.gpkg"