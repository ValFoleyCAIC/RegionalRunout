"""
Resource Monitor for Avalanche Pipeline

Author: Valerie Foley
Last Updated: 6/2026

Description:
    Tracks CPU, memory, disk I/O, and wall-clock time during pipeline
    runs. Runs as a background thread, sampling at configurable intervals,
    and generates a per-tile resource report plus scaling projections to
    help estimate hardware requirements for larger areas.
"""

import os
import time
import json
import threading
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)


# --------- System Info ---------

def get_system_info():
    # Collect static system information.
    # @returns: dict with CPU, memory, disk info

    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
        "cpu_count": os.cpu_count(),
        "python_version": None,
        "os": None,
    }

    import sys
    info["python_version"] = sys.version

    import platform
    info["os"] = f"{platform.system()} {platform.release()}"

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)

        disk = psutil.disk_usage("/")
        info["total_disk_gb"] = round(disk.total / (1024**3), 2)
        info["free_disk_gb"] = round(disk.free / (1024**3), 2)
    except ImportError:
        info["total_memory_gb"] = "psutil not installed"

    return info


# --------- Resource Sample ---------

@dataclass
class ResourceSample:
    timestamp: float
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    process_memory_gb: float = 0.0
    process_cpu_percent: float = 0.0
    active_threads: int = 0


def take_sample():
    # Take a single resource sample. Uses psutil if available, else falls
    # back to /proc and getloadavg.
    # @returns: ResourceSample

    sample = ResourceSample(timestamp=time.time())

    try:
        import psutil

        sample.cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        sample.memory_used_gb = round((mem.total - mem.available) / (1024**3), 3)
        sample.memory_percent = mem.percent

        try:
            disk_io = psutil.disk_io_counters()
            sample.disk_read_mb = round(disk_io.read_bytes / (1024**2), 1)
            sample.disk_write_mb = round(disk_io.write_bytes / (1024**2), 1)
        except (AttributeError, RuntimeError):
            pass

        proc = psutil.Process(os.getpid())
        proc_mem = proc.memory_info()
        sample.process_memory_gb = round(proc_mem.rss / (1024**3), 3)
        sample.process_cpu_percent = proc.cpu_percent(interval=None)
        sample.active_threads = proc.num_threads()

    except ImportError:
        # Fallback without psutil
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
                total = meminfo.get("MemTotal", 0) / (1024**2)
                available = meminfo.get("MemAvailable", 0) / (1024**2)
                sample.memory_used_gb = round(total - available, 3)
                sample.memory_percent = round(100 * (total - available) / total, 1) if total > 0 else 0
        except (FileNotFoundError, KeyError):
            pass

        try:
            load = os.getloadavg()
            cpus = os.cpu_count() or 1
            sample.cpu_percent = round(100 * load[0] / cpus, 1)
        except (OSError, AttributeError):
            pass

    return sample


# --------- Tile Timer ---------

@dataclass
class TileMetrics:
    tile_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_gb: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    dem_pixels: int = 0
    release_pixels: int = 0
    status: str = "pending"
    error: str = ""
    samples: List[Dict] = field(default_factory=list)


# --------- Monitor Thread ---------

class ResourceMonitor:
    # Background thread that samples system resources at regular intervals,
    # tracks per-tile metrics, and generates scaling reports.

    def __init__(self, output_dir, interval_seconds=5.0):
        # @param output_dir: Directory for monitor output files
        # @param interval_seconds: Sampling interval

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval_seconds

        self.system_info = get_system_info()
        self.samples = []
        self.tile_metrics = []
        self.current_tile = None

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Baseline disk I/O
        baseline = take_sample()
        self._baseline_disk_read = baseline.disk_read_mb
        self._baseline_disk_write = baseline.disk_write_mb

    def start(self):
        # Start background monitoring thread.
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"Resource monitor started (interval: {self.interval}s)")

    def stop(self):
        # Stop monitoring and write final report.
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)

        self._write_report()
        logger.info("Resource monitor stopped")

    def tile_start(self, tile_name, dem_pixels=0, release_pixels=0):
        # Note start of processing a tile.
        # @param tile_name: Name of the tile
        # @param dem_pixels: Total DEM pixels
        # @param release_pixels: Number of release area pixels

        with self._lock:
            # Close previous tile if still open
            if self.current_tile is not None:
                self._finalize_tile("complete")

            self.current_tile = TileMetrics(
                tile_name=tile_name,
                start_time=time.time(),
                dem_pixels=dem_pixels,
                release_pixels=release_pixels,
                status="running"
            )

        logger.debug(f"Monitor: tile started - {tile_name}")

    def tile_end(self, status="complete", error=""):
        # Note end of processing a tile.
        # @param status: Final status (complete/failed/skipped)
        # @param error: Error message if failed

        with self._lock:
            self._finalize_tile(status, error)

        logger.debug(f"Monitor: tile ended - {status}")

    def _finalize_tile(self, status, error=""):
        # Finalize metrics for the current tile.

        if self.current_tile is None:
            return

        tile = self.current_tile
        tile.end_time = time.time()
        tile.duration_seconds = round(tile.end_time - tile.start_time, 2)
        tile.status = status
        tile.error = error

        tile_samples = [s for s in self.samples
                        if tile.start_time <= s.timestamp <= tile.end_time]

        if tile_samples:
            tile.peak_memory_gb = max(s.memory_used_gb for s in tile_samples)
            tile.peak_cpu_percent = max(s.cpu_percent for s in tile_samples)
            tile.avg_cpu_percent = round(
                sum(s.cpu_percent for s in tile_samples) / len(tile_samples), 1
            )
            tile.avg_memory_gb = round(
                sum(s.memory_used_gb for s in tile_samples) / len(tile_samples), 3
            )

            first = tile_samples[0]
            last = tile_samples[-1]
            tile.disk_read_mb = round(last.disk_read_mb - first.disk_read_mb, 1)
            tile.disk_write_mb = round(last.disk_write_mb - first.disk_write_mb, 1)

            tile.samples = [
                {"t": round(s.timestamp - tile.start_time, 1),
                 "cpu": s.cpu_percent,
                 "mem_gb": s.memory_used_gb}
                for s in tile_samples
            ]

        self.tile_metrics.append(tile)
        self.current_tile = None

    def _sample_loop(self):
        # Background sampling loop.
        while self._running:
            sample = take_sample()
            self.samples.append(sample)
            time.sleep(self.interval)

    def _write_report(self):
        # Generate JSON + text resource reports.

        tiles_data = []
        for t in self.tile_metrics:
            tiles_data.append({
                "tile_name": t.tile_name,
                "status": t.status,
                "duration_seconds": t.duration_seconds,
                "duration_minutes": round(t.duration_seconds / 60, 2),
                "peak_memory_gb": t.peak_memory_gb,
                "avg_memory_gb": t.avg_memory_gb,
                "peak_cpu_percent": t.peak_cpu_percent,
                "avg_cpu_percent": t.avg_cpu_percent,
                "disk_read_mb": t.disk_read_mb,
                "disk_write_mb": t.disk_write_mb,
                "dem_pixels": t.dem_pixels,
                "release_pixels": t.release_pixels,
                "error": t.error,
            })

        completed_tiles = [t for t in self.tile_metrics if t.status == "complete"]
        scaling = {}

        if completed_tiles:
            durations = [t.duration_seconds for t in completed_tiles]
            peak_mems = [t.peak_memory_gb for t in completed_tiles]

            scaling = {
                "tiles_completed": len(completed_tiles),
                "total_runtime_minutes": round(sum(durations) / 60, 2),
                "avg_tile_minutes": round((sum(durations) / len(durations)) / 60, 2),
                "max_tile_minutes": round(max(durations) / 60, 2),
                "min_tile_minutes": round(min(durations) / 60, 2),
                "peak_memory_gb": round(max(peak_mems), 2),
                "avg_peak_memory_gb": round(sum(peak_mems) / len(peak_mems), 2),
            }

            # Projections for larger runs
            avg_minutes = sum(durations) / len(durations) / 60
            for n_tiles in [10, 25, 50, 100, 200]:
                hours = (n_tiles * avg_minutes) / 60
                scaling[f"estimated_{n_tiles}_tiles_hours"] = round(hours, 1)

            scaling["memory_recommendation"] = (
                f"Peak per tile: {max(peak_mems):.1f} GB. "
                f"For parallel processing with N workers, provision "
                f"~{max(peak_mems) * 1.2:.1f} GB per worker."
            )

        report = {
            "system_info": self.system_info,
            "monitoring": {
                "interval_seconds": self.interval,
                "total_samples": len(self.samples),
                "total_runtime_seconds": round(
                    self.samples[-1].timestamp - self.samples[0].timestamp, 2
                ) if len(self.samples) > 1 else 0,
            },
            "tile_metrics": tiles_data,
            "scaling_estimates": scaling,
        }

        report_path = self.output_dir / "resource_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Resource report: {report_path}")

        summary_path = self.output_dir / "resource_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("RESOURCE MONITORING SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("System:\n")
            f.write(f"  CPUs:   {self.system_info.get('cpu_count', '?')}\n")
            f.write(f"  Memory: {self.system_info.get('total_memory_gb', '?')} GB\n")
            f.write(f"  Disk:   {self.system_info.get('free_disk_gb', '?')} GB free\n\n")

            if completed_tiles:
                f.write(f"Tiles completed: {len(completed_tiles)}\n")
                f.write(f"Total runtime:   {scaling['total_runtime_minutes']:.1f} min\n")
                f.write(f"Avg per tile:    {scaling['avg_tile_minutes']:.1f} min\n")
                f.write(f"Max per tile:    {scaling['max_tile_minutes']:.1f} min\n")
                f.write(f"Peak memory:     {scaling['peak_memory_gb']:.1f} GB\n\n")

                f.write("-" * 70 + "\n")
                f.write("Per-tile breakdown:\n")
                f.write(f"{'Tile':<30} {'Time (min)':>10} {'Peak Mem (GB)':>14} "
                        f"{'Avg CPU %':>10} {'Status':>10}\n")
                f.write("-" * 70 + "\n")

                for t in tiles_data:
                    f.write(f"{t['tile_name']:<30} "
                            f"{t['duration_minutes']:>10.1f} "
                            f"{t['peak_memory_gb']:>14.1f} "
                            f"{t['avg_cpu_percent']:>10.1f} "
                            f"{t['status']:>10}\n")

                f.write("\n" + "-" * 70 + "\n")
                f.write("Scaling projections (sequential processing):\n")
                for n_tiles in [10, 25, 50, 100, 200]:
                    key = f"estimated_{n_tiles}_tiles_hours"
                    if key in scaling:
                        f.write(f"  {n_tiles:>3} tiles: ~{scaling[key]:.1f} hours\n")

                f.write(f"\n{scaling.get('memory_recommendation', '')}\n")

            failed = [t for t in self.tile_metrics if t.status == "failed"]
            if failed:
                f.write(f"\nFailed tiles: {len(failed)}\n")
                for t in failed:
                    f.write(f"  {t.tile_name}: {t.error[:100]}\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Resource summary: {summary_path}")


# --------- Standalone Mode ---------

def monitor_standalone(output_dir, interval=5.0, duration=None):
    # Run resource monitor standalone (for watching an external process).
    # @param output_dir: Output directory for reports
    # @param interval: Sampling interval in seconds
    # @param duration: Max duration in seconds (None = until interrupted)

    monitor = ResourceMonitor(output_dir, interval)
    monitor.start()

    print(f"Resource monitor running (interval: {interval}s)")
    print(f"Output: {output_dir}")
    print("Press Ctrl+C to stop and generate report...\n")

    try:
        start = time.time()
        while True:
            elapsed = time.time() - start
            sample = monitor.samples[-1] if monitor.samples else None

            if sample:
                print(f"\r  [{elapsed:.0f}s] "
                      f"CPU: {sample.cpu_percent:.0f}% | "
                      f"Mem: {sample.memory_used_gb:.1f} GB ({sample.memory_percent:.0f}%) | "
                      f"Threads: {sample.active_threads}",
                      end="", flush=True)

            if duration and elapsed >= duration:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n")

    monitor.stop()
    print("Report generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resource Monitor for Avalanche Pipeline")
    parser.add_argument("--output-dir", type=str, default="./resource_logs",
                        help="Output directory for reports")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Sampling interval in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max monitoring duration in seconds")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    monitor_standalone(args.output_dir, args.interval, args.duration)
