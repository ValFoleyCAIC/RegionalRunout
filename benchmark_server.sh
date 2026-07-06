#!/usr/bin/env bash
# benchmark_server.sh
#
# Run on your real server BEFORE setting anything up. Tells you:
#   - whether you have headroom for 18 x ~1m DEM tiles
#   - what to set --target-cpu-fraction / --min-free-mem-gb to
#   - whether other users are heavy right now
#
# Usage:
#   chmod +x benchmark_server.sh
#   ./benchmark_server.sh                    # one snapshot
#   ./benchmark_server.sh --watch 10         # snapshot every 10s for 5 min
#                                            # (catches load swings from other users)

set -u

WATCH_S=0
SAMPLES=30
if [[ "${1:-}" == "--watch" ]]; then
    WATCH_S="${2:-10}"
fi

snapshot() {
    echo "=================================================================="
    echo "Snapshot at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "=================================================================="

    echo
    echo "--- Host ---"
    hostname
    uname -srm

    echo
    echo "--- CPU ---"
    local ncpu
    ncpu=$(nproc)
    echo "CPUs (logical): $ncpu"
    lscpu 2>/dev/null | grep -E "^Model name|^Socket|^Core|^Thread" | sed 's/^/  /'

    echo
    echo "--- Load ---"
    # Load avg + per-CPU ratio (this is what adaptive_throttle gates on)
    read -r l1 l5 l15 _ < /proc/loadavg
    awk -v l1="$l1" -v n="$ncpu" 'BEGIN {
        printf "  load: 1m=%.2f  5m='"$l5"'  15m='"$l15"'\n", l1
        printf "  load_per_cpu (1m): %.0f%% of %d CPUs\n", 100*l1/n, n
        if (l1/n > 0.85) print "  WARNING: per-CPU load > 85% — your throttle would currently BLOCK"
    }'

    echo
    echo "--- Memory ---"
    free -h | sed 's/^/  /'
    echo
    awk '/^MemTotal:|^MemAvailable:/ {print "  " $0}' /proc/meminfo
    awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2}
         END {
             printf "  used: %.1f GB of %.1f GB (%.0f%%), available: %.1f GB\n", \
                 (t-a)/1024/1024, t/1024/1024, 100*(t-a)/t, a/1024/1024
         }' /proc/meminfo

    echo
    echo "--- Disk (project dir + output dir candidates) ---"
    df -h /home 2>/dev/null | sed 's/^/  /'
    df -h /mnt 2>/dev/null | grep -v "^Filesystem" | sed 's/^/  /' || true

    echo
    echo "--- Top non-system memory hogs (other users / processes) ---"
    # Pick the top 5 RSS users that aren't kernel/system
    ps -eo user,pid,pcpu,pmem,rss,comm --sort=-rss \
        | awk 'NR==1 || ($1!="root" && $1!="systemd+" && $5>50000)' \
        | head -6 \
        | sed 's/^/  /'

    echo
    echo "--- Active users on box ---"
    who | awk '{print $1}' | sort -u | sed 's/^/  /' || echo "  (none)"

    echo
    echo "--- Python / GDAL availability ---"
    for cmd in python3 conda mamba gdalinfo; do
        if command -v "$cmd" >/dev/null 2>&1; then
            printf "  %-10s %s\n" "$cmd:" "$($cmd --version 2>&1 | head -1)"
        else
            printf "  %-10s NOT FOUND\n" "$cmd:"
        fi
    done
}

if [[ "$WATCH_S" -gt 0 ]]; then
    echo "Watching every ${WATCH_S}s for ${SAMPLES} samples (~$((WATCH_S*SAMPLES/60)) min)"
    echo "This catches load swings caused by other users."
    echo
    for i in $(seq 1 "$SAMPLES"); do
        echo
        echo "### Sample $i / $SAMPLES ###"
        # Compact form during watch — full form only on first/last
        if [[ "$i" == "1" || "$i" == "$SAMPLES" ]]; then
            snapshot
        else
            read -r l1 _ < /proc/loadavg
            ncpu=$(nproc)
            avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
            avail_gb=$(awk -v k="$avail_kb" 'BEGIN{printf "%.1f", k/1024/1024}')
            printf "  %s  load=%s  load/cpu=%.0f%%  mem_avail=%sGB\n" \
                "$(date '+%H:%M:%S')" "$l1" \
                "$(awk -v l="$l1" -v n="$ncpu" 'BEGIN{print 100*l/n}')" \
                "$avail_gb"
        fi
        sleep "$WATCH_S"
    done
else
    snapshot

    echo
    echo "=================================================================="
    echo "Sizing recommendation for regionalRunout (18 features, 1m DEMs)"
    echo "=================================================================="
    awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2}
         END {
             tg = t/1024/1024; ag = a/1024/1024
             printf "  total RAM:       %.0f GB\n", tg
             printf "  available now:   %.0f GB\n", ag
             # 1m DEM, ~5x5 km tile -> ~25M pixels float32 = ~100 MB raw,
             # plus FlowPy peak ~5-10x for buffered tile + intermediates.
             # Budget ~16 GB per worker is what your config defaults to.
             printf "  workers fitting in 25%% of total: %d (at 16 GB each)\n", \
                 int(tg*0.25/16)
             printf "  workers fitting in available now: %d (at 16 GB each)\n", \
                 int(ag/16)
             if (ag < 32) print "  WARNING: <32 GB available — your default --min-free-mem-gb=32 will pause the run"
         }' /proc/meminfo

    echo
    echo "Run again with '--watch 30' over a typical busy period to see"
    echo "if other users push load_per_cpu above 85% (your throttle blocks there)."
fi
