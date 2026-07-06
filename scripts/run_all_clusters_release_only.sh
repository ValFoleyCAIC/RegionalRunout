#!/bin/bash
# Release-only run across all clusters.
# No time window, no mosaic - just produces per-tile release polygons
# you can browse and select from for downstream FlowPy runs.

set -u
cd /home/val/regionalRunout/pipeline

LOGDIR=/home/val/regionalRunout/logs
DEMDIR=/home/val/regionalRunout/data/DEM
OUTDIR=/home/val/regionalRunout/outputs_release_only

CLUSTERS="${CLUSTERS:-11 06 07 02 14 03 05 15 08 01 17 00}"

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Release-only run across all clusters"
echo "============================================================"

for cluster_num in $CLUSTERS; do
    CLUSTER="cluster_${cluster_num}"
    DEM_PATH="${DEMDIR}/${CLUSTER}.tif"

    if [ ! -f "$DEM_PATH" ]; then
        echo "$(date '+%H:%M:%S') $CLUSTER: DEM not found - skipping"
        continue
    fi

    echo
    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Starting $CLUSTER"
    echo "============================================================"

    TILE_IDS=$(python -c "
import rasterio, sys
sys.path.insert(0, '/home/val/regionalRunout/pipeline')
from tile_processor.tile_grid import compute_tile_grid

with rasterio.open('$DEM_PATH') as src:
    bbox = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

tiles = compute_tile_grid(bbox, core_m=9000.0, overlap_m=3000.0)
for t in tiles:
    print(t.tile_id)
")
    if [ -z "$TILE_IDS" ]; then
        echo "$(date '+%H:%M:%S') $CLUSTER: failed to compute tile grid - skipping"
        continue
    fi

    N_TILES=$(echo "$TILE_IDS" | wc -l)
    echo "  $N_TILES tiles to process"

    TILE_NUM=0
    for tile in $TILE_IDS; do
        TILE_NUM=$((TILE_NUM + 1))

        LOG="$LOGDIR/${CLUSTER}_tile_${tile}_release.log"
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $TILE_NUM/$N_TILES ($tile) starting"

        python -m tile_processor.tile_runner \
            --cluster-id "$CLUSTER" \
            --tile-id "$tile" \
            --mode release-only \
            --output-dir "$OUTDIR" \
            > "$LOG" 2>&1

        if grep -q "status=complete" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: complete"
        elif grep -q "status=skipped" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: skipped (already done)"
        else
            echo "$(date '+%H:%M:%S')   tile $tile: status unclear (see $LOG)"
        fi
    done
done

echo
echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') ALL CLUSTERS COMPLETE"
echo "============================================================"
echo "Release polygons in: $OUTDIR/<cluster>/tiles/tile_<row>_<col>/release_areas/"
