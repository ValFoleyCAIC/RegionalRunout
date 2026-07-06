#!/bin/bash
# Snapshot of the flowpy-only selected-releases run.
OUTDIR=/home/val/regionalRunout/outputs_flowpy_selected

echo "=== Flowpy-only progress ==="
echo

DONE=$(find "$OUTDIR" -name "*_summary.json" \
    -exec grep -l '"status": "complete"' {} \; 2>/dev/null | wc -l)
echo "Tiles complete: $DONE"
echo

echo "Per-cluster:"
for c in 11 06 07 02 14 03 05 15 08 01 17 00; do
    CDIR="$OUTDIR/cluster_${c}"
    if [ -d "$CDIR" ]; then
        N=$(find "$CDIR/tiles" -name "*_summary.json" \
            -exec grep -l '"status": "complete"' {} \; 2>/dev/null | wc -l)
        TOTAL=$(ls -d "$CDIR/tiles"/tile_* 2>/dev/null | wc -l)
        echo "  cluster_${c}: $N / $TOTAL"
    fi
done
echo

echo "Last 5 finished:"
find "$OUTDIR" -name "*_summary.json" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -5 \
    | awk '{cmd="date -d @"$1" \"+%Y-%m-%d %H:%M\""; cmd | getline ts; close(cmd); print "  "ts"  "$2}'
echo

if pgrep -f "tile_processor.tile_runner" > /dev/null; then
    NPROCS=$(pgrep -fc 'tile_processor.tile_runner')
    NFLOW=$(pgrep -fc 'com4FlowPy' 2>/dev/null || echo 0)
    echo "Status: RUNNING ($NPROCS tile_runner, $NFLOW FlowPy worker)"
else
    echo "Status: not running"
fi
