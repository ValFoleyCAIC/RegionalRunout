"""
Schedule Runner for Avalanche Pipeline

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Supervisor that runs the pipeline only during a configured time window
    (default 12:00 PM - 4:00 AM Mountain Time). Graceful cutoff: when the
    window closes it writes a stop file (<output_dir>/STOP_AFTER_TILE) that
    main.py checks between tiles, so the current tile finishes and the
    worker exits cleanly, then resumes the next day. Skip-resume relies on
    the pipeline's own per-tile checkpoints. DST handled via zoneinfo.
    Safe under nohup for weeks.
    
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# --------- Window Logic ---------

def parse_hhmm(s: str) -> tuple[int, int]:
    # Parse "HH:MM" -> (hour, minute), validating ranges.
    # @param s: time string "HH:MM"
    # @returns: (hour, minute) tuple
    h, m = s.split(":")
    h, m = int(h), int(m)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"Invalid time {s}")
    return h, m


def in_window(now: datetime, start_hm: tuple[int, int],
              end_hm: tuple[int, int]) -> bool:
    # Test whether `now` falls within [start_hm, end_hm], handling windows
    # that cross midnight (e.g. 12:00 -> 04:00 next day).
    # @param now: current datetime
    # @param start_hm: window open (hour, minute)
    # @param end_hm: window close (hour, minute)
    # @returns: True if inside the window
    sh, sm = start_hm
    eh, em = end_hm

    minutes_now = now.hour * 60 + now.minute
    minutes_start = sh * 60 + sm
    minutes_end = eh * 60 + em

    if minutes_start <= minutes_end:
        return minutes_start <= minutes_now < minutes_end
    else:
        # Crosses midnight
        return minutes_now >= minutes_start or minutes_now < minutes_end


def seconds_until_next_open(now: datetime, start_hm: tuple[int, int]) -> float:
    # Seconds until the next occurrence of start_hm.
    # @param now: current datetime
    # @param start_hm: window open (hour, minute)
    # @returns: seconds until next open
    sh, sm = start_hm
    target = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


# --------- Supervisor ---------

class Supervisor:
    # Manages the pipeline subprocess across day/night cycles.

    def __init__(
        self,
        worker_cmd: list[str],
        output_dir: Path,
        window_start: tuple[int, int],
        window_end: tuple[int, int],
        tz: ZoneInfo,
        check_interval_s: float = 60.0,
        cutoff_grace_s: float = 4 * 3600.0,  # max wait for graceful exit
    ):
        self.worker_cmd = worker_cmd
        self.output_dir = Path(output_dir)
        self.window_start = window_start
        self.window_end = window_end
        self.tz = tz
        self.check_interval_s = check_interval_s
        self.cutoff_grace_s = cutoff_grace_s

        self.stop_file = self.output_dir / "STOP_AFTER_TILE"
        self.proc: subprocess.Popen | None = None
        self._shutdown_requested = False

        # Catch SIGTERM/SIGINT to shut down the worker cleanly
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum, _frame):
        logger.info(f"Supervisor received signal {signum}, shutting down")
        self._shutdown_requested = True

    def _now(self) -> datetime:
        return datetime.now(self.tz)

    def _ensure_no_stop_file(self) -> None:
        # Remove any leftover stop flag from a previous cycle.
        if self.stop_file.exists():
            try:
                self.stop_file.unlink()
                logger.info(f"Removed stale stop file: {self.stop_file}")
            except OSError as e:
                logger.warning(f"Could not remove stop file: {e}")

    def _write_stop_file(self, reason: str) -> None:
        # Signal to the pipeline: finish current tile, then exit.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text(
            f"Stop requested at {self._now().isoformat()}\nReason: {reason}\n"
        )
        logger.info(f"Wrote stop file: {self.stop_file} ({reason})")

    def _start_worker(self) -> None:
        # Launch the pipeline as a subprocess. start_new_session lets us
        # signal the whole process group if needed.
        self._ensure_no_stop_file()

        logger.info(f"Starting worker: {' '.join(self.worker_cmd)}")
        self.proc = subprocess.Popen(
            self.worker_cmd,
            start_new_session=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        logger.info(f"Worker started, PID={self.proc.pid}")

    def _worker_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _wait_for_worker_exit(self, timeout_s: float) -> bool:
        # Wait up to timeout_s for the worker to exit.
        # @returns: True if it exited
        if self.proc is None:
            return True
        try:
            self.proc.wait(timeout=timeout_s)
            logger.info(f"Worker exited with code {self.proc.returncode}")
            return True
        except subprocess.TimeoutExpired:
            return False

    def _hard_terminate(self) -> None:
        # Last resort: kill the worker process group.
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            logger.warning(f"Hard-terminating worker process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(5)
            if self._worker_running():
                logger.warning("Worker did not exit on SIGTERM, sending SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run_forever(self) -> None:
        # Main supervisor loop.
        logger.info("=" * 70)
        logger.info("AVALANCHE PIPELINE SUPERVISOR")
        logger.info(f"  Window:  {self.window_start[0]:02d}:{self.window_start[1]:02d} "
                    f"-> {self.window_end[0]:02d}:{self.window_end[1]:02d} "
                    f"({self.tz.key})")
        logger.info(f"  Output:  {self.output_dir}")
        logger.info(f"  Worker:  {' '.join(self.worker_cmd)}")
        logger.info("=" * 70)

        while not self._shutdown_requested:
            now = self._now()
            inside = in_window(now, self.window_start, self.window_end)

            if inside and not self._worker_running():
                # Window open, no worker -> start one
                self._start_worker()

            elif inside and self._worker_running():
                # Window open, worker running -> check if it finished on its own
                if self.proc.poll() is not None:
                    code = self.proc.returncode
                    logger.info(f"Worker exited cleanly with code {code}")
                    if code == 0:
                        logger.info("All tiles complete — supervisor will idle")
                    self.proc = None

            elif not inside and self._worker_running():
                # Window closed, worker running -> request graceful stop
                logger.info(
                    f"Window closed at {now.strftime('%H:%M %Z')} — "
                    f"requesting graceful stop after current tile"
                )
                self._write_stop_file("Window closed")

                # Give the worker up to cutoff_grace_s to finish current tile
                t0 = time.time()
                while self._worker_running() and (time.time() - t0) < self.cutoff_grace_s:
                    if self._shutdown_requested:
                        break
                    elapsed = time.time() - t0
                    if int(elapsed) % 300 == 0:  # log every 5 min
                        logger.info(
                            f"Waiting for worker to finish current tile "
                            f"({elapsed:.0f}s elapsed, "
                            f"{self.cutoff_grace_s - elapsed:.0f}s remaining)"
                        )
                    time.sleep(30)

                if self._worker_running():
                    logger.warning(
                        f"Worker did not exit within {self.cutoff_grace_s}s grace — "
                        f"hard-terminating to free resources"
                    )
                    self._hard_terminate()
                    self._wait_for_worker_exit(timeout_s=30)

                self.proc = None
                self._ensure_no_stop_file()

            elif not inside and not self._worker_running():
                # Window closed, no worker -> sleep until next open
                wait_s = seconds_until_next_open(now, self.window_start)
                next_open = now + timedelta(seconds=wait_s)
                logger.info(
                    f"Outside window. Next open: "
                    f"{next_open.strftime('%Y-%m-%d %H:%M %Z')} "
                    f"(sleeping {wait_s/3600:.1f}h)"
                )
                # Sleep in chunks 
                end_t = time.time() + wait_s
                while time.time() < end_t and not self._shutdown_requested:
                    time.sleep(min(60.0, end_t - time.time()))
                continue

            time.sleep(self.check_interval_s)

        # Shutdown: stop worker if running
        if self._worker_running():
            logger.info("Supervisor shutting down — stopping worker")
            self._write_stop_file("Supervisor shutdown")
            if not self._wait_for_worker_exit(timeout_s=300):
                self._hard_terminate()
                self._wait_for_worker_exit(timeout_s=30)

        logger.info("Supervisor exited cleanly")


# --------- CLI ---------

def main():
    p = argparse.ArgumentParser(
        description="Supervisor for windowed avalanche pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--worker-cmd",
        type=str,
        default="python main.py --monitor",
        help="Command to run the pipeline (default: 'python main.py --monitor')",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Pipeline output directory (must match main.py's --output-dir; "
             "supervisor writes STOP_AFTER_TILE here)",
    )
    p.add_argument("--window-start", type=str, default="12:00",
                   help="Window open time HH:MM (default: 12:00)")
    p.add_argument("--window-end", type=str, default="04:00",
                   help="Window close time HH:MM (default: 04:00 next day)")
    p.add_argument("--tz", type=str, default="America/Denver",
                   help="Timezone (default: America/Denver - Mountain Time)")
    p.add_argument("--check-interval", type=float, default=60.0,
                   help="Supervisor check interval, seconds (default: 60)")
    p.add_argument("--cutoff-grace", type=float, default=4.0,
                   help="Max hours to wait for current tile to finish "
                        "after window closes (default: 4.0)")

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [supervisor] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sup = Supervisor(
        worker_cmd=args.worker_cmd.split(),
        output_dir=args.output_dir,
        window_start=parse_hhmm(args.window_start),
        window_end=parse_hhmm(args.window_end),
        tz=ZoneInfo(args.tz),
        check_interval_s=args.check_interval,
        cutoff_grace_s=args.cutoff_grace * 3600.0,
    )
    sup.run_forever()


if __name__ == "__main__":
    main()
