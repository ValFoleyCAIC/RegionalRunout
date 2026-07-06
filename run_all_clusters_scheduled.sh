#!/bin/bash
# Scheduled multi-cluster runner with auto-mosaic per cluster.
#
# Time window: runs 11am to 4am next day MDT.
# Block window: 4am to 11am MDT (4am cutoff protects against
# tiles bleeding into 7am critical process).
#
# Behavior:
#   - Before processing the next tile, check current hour
#   - If we're in the block window (4-11am), sleep until 11:00
#   - Between tiles is the natural pause point (~150 min per tile,
#     no mid-tile interruption — that would waste 2-3 hours)
#   - After all tiles for a cluster finish, auto-run the mosaicker
#   - Skip-resume via tile_runner's is_tile_done

set -u
cd /home/val/regionalRunout/pipeline

LOGDIR=/home/val/regionalRunout/logs
DEMDIR=/home/val/regionalRunout/data/DEM

# Process in size order (smallest first for fast feedback)
CLUSTERS="11 06 07 02 14 03 05 15 08 01 17 00"

# --------- Helpers ---------

# Returns 0 if we're in the block window (4:00-10:59 MDT), 1 otherwise.
in_block_window() {
    local hour
    hour=$(TZ=America/Denver date +%H)
    if [ "$hour" -ge 4 ] && [ "$hour" -lt 11 ]; then
        return 0
    fi
    return 1
}

# Seconds until 11:00 AM MDT (next occurrence).
seconds_until_11am() {
    local now_epoch eleven_epoch
    now_epoch=$(date +%s)
    eleven_epoch=$(TZ=America/Denver date -d "today 11:00" +%s)
    if [ "$eleven_epoch" -le "$now_epoch" ]; then
        eleven_epoch=$(TZ=America/Denver date -d "tomorrow 11:00" +%s)
    fi
    echo $((eleven_epoch - now_epoch))
}

# Sleep until the run window opens. Logs every 30 min while sleeping
# so we know it's still alive.
wait_for_window() {
    while in_block_window; do
        local secs
        secs=$(seconds_until_11am)
        local mins=$((secs / 60))
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z') BLOCK WINDOW (4-11am MDT) - sleeping $mins min"
        if [ "$secs" -gt 1800 ]; then
            sleep 1800
        else
            sleep "$secs"
        fi
    done
}

# --------- Main Loop ---------

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Scheduled multi-cluster runner starting"
echo "============================================================"

wait_for_window
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') In run window - starting work"

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
    CLUSTER_FAILED=0
    for tile in $TILE_IDS; do
        TILE_NUM=$((TILE_NUM + 1))

        wait_for_window

        LOG="$LOGDIR/${CLUSTER}_tile_${tile}.log"
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $TILE_NUM/$N_TILES ($tile) starting"

        python -m tile_processor.tile_runner \
            --cluster-id "$CLUSTER" \
            --tile-id "$tile" \
            --output-dir /home/val/regionalRunout/outputs_release_1m \
            > "$LOG" 2>&1

        if grep -q "WATCHDOG TRIGGERED" "$LOG" 2>/dev/null; then
            echo "$(date '+%H:%M:%S')   WARNING: watchdog killed tile $tile"
            echo "$(date '+%H:%M:%S')   aborting $CLUSTER, continuing to next cluster"
            CLUSTER_FAILED=1
            break
        fi
        if grep -q "status=complete" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: complete"
        elif grep -q "status=skipped" "$LOG" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   tile $tile: skipped (already done)"
        else
            echo "$(date '+%H:%M:%S')   tile $tile: status unclear (see log)"
        fi
    done

    if [ "$CLUSTER_FAILED" = "1" ]; then
        echo "$(date '+%H:%M:%S') $CLUSTER incomplete - skipping mosaic, moving on"
        continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   mosaicking $CLUSTER..."
    python -m tile_processor.raster_mosaicker \
        --cluster-id "$CLUSTER" \
        > "$LOGDIR/${CLUSTER}_mosaic.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   $CLUSTER mosaic complete"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z')   $CLUSTER mosaic FAILED (see ${LOGDIR}/${CLUSTER}_mosaic.log)"
    fi
done

echo
echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') ALL CLUSTERS COMPLETE"
echo "============================================================"
