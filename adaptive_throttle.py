"""
Adaptive Throttle for Shared-Server Avalanche Pipeline

Author: Valerie Foley
Last Updated: 4/2026

Description:
    Active resource throttling for the pipeline on a shared multi-CPU
    server. Complements resource_monitor.py (passive reporter): this is
    the active controller that waits and scales workers. Sets OS-level
    low priority once at process start (nice +19, ionice idle), then
    between DEM tiles blocks on high load / low RAM / low disk and scales
    the FlowPy CPU cap down when other users are busy. Decisions are made
    between tiles so a running simulation is never interrupted. Targets
    ~25% of total resources and always yields to other users.
"""

import os
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# --------- One-Time Process Setup ---------

def set_low_priority(nice_level: int = 19, io_class: int = 3) -> None:
    # Set process to lowest CPU and I/O priority on Linux. Best-effort:
    # logs but does not raise on failure.
    # @param nice_level: 19 = lowest (range -20 to 19)
    # @param io_class: 3 = idle (runs only when disk is otherwise idle)
    # @returns: None

    # CPU nice - works on any POSIX system
    try:
        current = os.nice(0)
        delta = nice_level - current
        if delta > 0:
            os.nice(delta)
            logger.info(f"Process nice level set to {nice_level} (was {current})")
        else:
            logger.info(f"Process nice level already at {current}, not lowering further")
    except Exception as e:
        logger.warning(f"Could not set nice level: {e}")

    # I/O priority - Linux only. Try the syscall first (no subprocess),
    # fall back to the ionice command.
    try:
        # ioprio_set: which=1 (process), who=0 (self), ioprio=(class<<13)|prio
        import ctypes
        IOPRIO_WHO_PROCESS = 1
        IOPRIO_CLASS_SHIFT = 13
        ioprio = (io_class << IOPRIO_CLASS_SHIFT) | 0
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.syscall(251, IOPRIO_WHO_PROCESS, 0, ioprio)  # 251 = x86_64
        if result == 0:
            logger.info(f"Process I/O priority set to class {io_class} (idle)")
        else:
            errno = ctypes.get_errno()
            raise OSError(errno, f"ioprio_set syscall failed (errno {errno})")
    except Exception as e:
        logger.debug(f"ioprio_set syscall failed ({e}), trying ionice subprocess")
        try:
            import subprocess
            pid = os.getpid()
            subprocess.run(
                ["ionice", "-c", str(io_class), "-p", str(pid)],
                check=True, capture_output=True
            )
            logger.info(f"Process I/O priority set via ionice to class {io_class}")
        except Exception as e2:
            logger.warning(f"Could not set I/O priority: {e2}")


# --------- Throttle State ---------

@dataclass
class SystemState:
    # Snapshot of system resources at one instant
    load_1min: float       # 1-minute load average
    load_per_cpu: float    # load_1min / cpu_count (1.0 = fully loaded)
    cpu_count: int
    mem_total_gb: float
    mem_avail_gb: float
    mem_used_pct: float
    free_disk_gb: float    # output disk free space


def read_system_state(output_disk: str = "/") -> SystemState:
    # Read current system state from /proc and os.statvfs (pure stdlib).
    # @param output_disk: Path to check disk free space
    # @returns: SystemState

    cpu_count = os.cpu_count() or 1

    try:
        load_1, _, _ = os.getloadavg()
    except (OSError, AttributeError):
        load_1 = 0.0

    mem_total_kb = 0
    mem_avail_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                key = parts[0].rstrip(":")
                if key == "MemTotal":
                    mem_total_kb = int(parts[1])
                elif key == "MemAvailable":
                    mem_avail_kb = int(parts[1])
                if mem_total_kb and mem_avail_kb:
                    break
    except (FileNotFoundError, IndexError, ValueError):
        pass

    mem_total_gb = mem_total_kb / (1024**2) if mem_total_kb else 0.0
    mem_avail_gb = mem_avail_kb / (1024**2) if mem_avail_kb else 0.0
    mem_used_pct = (
        100.0 * (mem_total_kb - mem_avail_kb) / mem_total_kb
        if mem_total_kb else 0.0
    )

    try:
        st = os.statvfs(output_disk)
        free_disk_gb = (st.f_bavail * st.f_frsize) / (1024**3)
    except (OSError, FileNotFoundError):
        free_disk_gb = 0.0

    return SystemState(
        load_1min=load_1,
        load_per_cpu=load_1 / cpu_count,
        cpu_count=cpu_count,
        mem_total_gb=mem_total_gb,
        mem_avail_gb=mem_avail_gb,
        mem_used_pct=mem_used_pct,
        free_disk_gb=free_disk_gb,
    )


# --------- Adaptive Throttle ---------

class AdaptiveThrottle:
    # Decides whether the pipeline should run, wait, or scale back based
    # on shared-server load. Polite (~25% target).

    def __init__(
        self,
        target_cpu_fraction: float = 0.25,
        target_mem_fraction: float = 0.25,
        max_load_per_cpu: float = 0.85,   # other users using >85% of CPUs
        min_free_mem_gb: float = 32.0,    # always leave 32 GB headroom
        min_free_disk_gb: float = 50.0,   # need 50 GB free for outputs
        flowpy_cpu_floor: int = 2,
        flowpy_cpu_ceiling: int = 8,
        check_interval_s: float = 30.0,
        max_wait_s: float = 3600.0,       # max 1 hour blocked before warning
        output_disk: str = "/",
    ):
        self.target_cpu_fraction = target_cpu_fraction
        self.target_mem_fraction = target_mem_fraction
        self.max_load_per_cpu = max_load_per_cpu
        self.min_free_mem_gb = min_free_mem_gb
        self.min_free_disk_gb = min_free_disk_gb
        self.flowpy_cpu_floor = flowpy_cpu_floor
        self.flowpy_cpu_ceiling = flowpy_cpu_ceiling
        self.check_interval_s = check_interval_s
        self.max_wait_s = max_wait_s
        self.output_disk = output_disk

        # Cache last state so other code can read without re-polling
        self.last_state: SystemState = read_system_state(output_disk)

    def snapshot(self) -> SystemState:
        # Refresh and return current state.
        self.last_state = read_system_state(self.output_disk)
        return self.last_state

    def is_ok_to_run(self, state: SystemState) -> tuple[bool, str]:
        # Check whether it's safe to start the next tile.
        # @param state: SystemState to evaluate
        # @returns: (ok, reason). reason is empty if ok.

        if state.free_disk_gb < self.min_free_disk_gb:
            return False, (
                f"Disk almost full: {state.free_disk_gb:.1f} GB free "
                f"< {self.min_free_disk_gb} GB threshold"
            )

        if state.mem_avail_gb < self.min_free_mem_gb:
            return False, (
                f"Memory tight: {state.mem_avail_gb:.1f} GB available "
                f"< {self.min_free_mem_gb} GB threshold"
            )

        if state.load_per_cpu > self.max_load_per_cpu:
            return False, (
                f"Load high: {state.load_1min:.1f} on {state.cpu_count} CPUs "
                f"({100*state.load_per_cpu:.0f}% > {100*self.max_load_per_cpu:.0f}%)"
            )

        return True, ""

    def wait_until_ok(self) -> SystemState:
        # Block until system state is acceptable, polling every
        # check_interval_s. Logs while waiting.
        # @returns: the SystemState that finally cleared the gate

        waited = 0.0
        last_reason = ""

        while True:
            state = self.snapshot()
            ok, reason = self.is_ok_to_run(state)

            if ok:
                if waited > 0:
                    logger.info(
                        f"Resources OK after {waited:.0f}s wait — proceeding "
                        f"(load={state.load_1min:.1f}, "
                        f"avail_mem={state.mem_avail_gb:.1f}GB)"
                    )
                return state

            # Log only when reason changes to avoid spam every 30s
            if reason != last_reason:
                logger.info(f"Throttle waiting: {reason}")
                last_reason = reason

            if waited >= self.max_wait_s:
                logger.warning(
                    f"Still waiting after {waited:.0f}s ({reason}). Continuing."
                )
                waited = 0.0  # warn once per max_wait_s window

            time.sleep(self.check_interval_s)
            waited += self.check_interval_s

    def recommended_cpu_cap(self, state: SystemState | None = None) -> int:
        # Recommend FlowPy CPU count given current load. Targets
        # ~target_cpu_fraction of total CPUs, yielding if others are busy.
        # @param state: SystemState, or None to snapshot fresh
        # @returns: int CPU count, clamped to [floor, ceiling]

        if state is None:
            state = self.snapshot()

        target = int(state.cpu_count * self.target_cpu_fraction)

        # load_1min ≈ runnable processes; treat as others' usage
        others_load = max(0.0, state.load_1min)
        free_cores = max(0, state.cpu_count - int(others_load))

        n_cpu = min(target, free_cores)
        n_cpu = max(self.flowpy_cpu_floor, min(self.flowpy_cpu_ceiling, n_cpu))

        return n_cpu

    def recommended_workers(self, state: SystemState | None = None,
                            mem_per_worker_gb: float = 16.0) -> int:
        # Recommend max parallel pipeline workers based on RAM headroom.
        # Always 1 in the current sequential pipeline; ready for future
        # multiprocessing.Pool use.
        # @param state: SystemState, or None to snapshot fresh
        # @param mem_per_worker_gb: RAM budget per worker
        # @returns: int worker count, >= 1

        if state is None:
            state = self.snapshot()

        budget_gb = max(
            0.0,
            state.mem_total_gb * self.target_mem_fraction - self.min_free_mem_gb / 4
        )
        n_workers = max(1, int(budget_gb / mem_per_worker_gb))
        return n_workers

    def status_line(self, state: SystemState | None = None) -> str:
        # Compact one-line status for logging.
        # @param state: SystemState, or None to use last cached
        # @returns: status string
        if state is None:
            state = self.last_state
        return (
            f"load={state.load_1min:.1f}/{state.cpu_count} "
            f"({100*state.load_per_cpu:.0f}%) | "
            f"mem={state.mem_avail_gb:.1f}GB free "
            f"({state.mem_used_pct:.0f}% used) | "
            f"disk={state.free_disk_gb:.0f}GB free | "
            f"flowpy_cpu={self.recommended_cpu_cap(state)}"
        )


# --------- Smoke Test ---------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("Setting low priority...")
    set_low_priority()

    print("\nReading current system state:")
    state = read_system_state()
    for k, v in state.__dict__.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.2f}")
        else:
            print(f"  {k:20s} = {v}")

    print("\nThrottle decision:")
    throttle = AdaptiveThrottle()
    ok, reason = throttle.is_ok_to_run(state)
    print(f"  ok_to_run: {ok}")
    if reason:
        print(f"  reason:    {reason}")
    print(f"  cpu_cap:   {throttle.recommended_cpu_cap(state)}")
    print(f"  workers:   {throttle.recommended_workers(state)}")
    print(f"\n  {throttle.status_line(state)}")
