"""Constrain the edge process to a smaller computer (Windows Job Object).

The edge node simulates the UAV's onboard computer. Its limits are applied to
its own process through one Windows Job Object -- the mechanism behind
Docker's ``--cpus`` on Windows containers, and the counterpart of the Linux
cgroup CFS quota:

* **CPU budget** (``cpu_cores``): a hard cap on CPU time, expressed in cores'
  worth (e.g. 1.5). Measured to hold within ~15% (2026-10-12).
* **Which cores** (``cpus``): affinity, e.g. the i9-14900HX's efficiency
  cores (logical CPUs 16-31) as the closer analogue of an embedded SoC.
* **Memory** (``memory_mb``): commit limit; exceeding it fails allocations.
* **Threads**: torch/OpenCV pools sized to the budget, so the process does
  not oversubscribe its own cap.
* **Device** is chosen by the caller (``--device cpu`` for a CPU-only edge).
  A GPU edge cannot be throttled per process on Windows, so it is only an
  upper bound.

Limits a simulation must state when reporting results:

* A capped x86 core is not an ARM core -- report these as *compute budgets*,
  not as a named device, unless calibrated against real hardware.
* A hard cap is enforced by duty-cycling: average throughput is right, but
  work runs in bursts, so latency jitter is higher than on a genuinely slower
  chip. Docker's ``--cpus`` behaves the same.

Child processes **break away** from the job (``SILENT_BREAKAWAY_OK``), so the
simulated camera, which stands in for separate camera hardware, is never
charged to the edge's budget.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

_JOB_HANDLE = None  # kept open for the life of the process (queries, and to be explicit)


@dataclass(frozen=True)
class EdgeHardware:
    cpu_cores: float | None = None
    cpus: tuple[int, ...] | None = None
    memory_mb: int | None = None
    threads: int | None = None

    def thread_count(self) -> int | None:
        if self.threads is not None:
            return self.threads
        candidates = []
        if self.cpus:
            candidates.append(len(self.cpus))
        if self.cpu_cores is not None:
            candidates.append(max(1, int(-(-self.cpu_cores // 1))))  # ceil
        return min(candidates) if candidates else None

    @property
    def constrained(self) -> bool:
        return any(v is not None for v in (self.cpu_cores, self.cpus, self.memory_mb, self.threads))

    def describe(self) -> str:
        if not self.constrained:
            return "unconstrained (whole laptop)"
        parts = []
        if self.cpu_cores is not None:
            parts.append(f"CPU cap {self.cpu_cores:g} cores")
        if self.cpus:
            parts.append(f"on CPUs {format_cpu_list(self.cpus)}")
        if self.thread_count() is not None:
            parts.append(f"{self.thread_count()} threads")
        if self.memory_mb is not None:
            parts.append(f"memory {self.memory_mb} MB")
        return ", ".join(parts)


def parse_cpu_list(text: str | None) -> tuple[int, ...] | None:
    """``"16-19,24"`` -> ``(16, 17, 18, 19, 24)``."""

    if not text:
        return None
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-"))
            cpus.extend(range(lo, hi + 1))
        elif part:
            cpus.append(int(part))
    n = os.cpu_count() or 1
    bad = [c for c in cpus if not 0 <= c < n]
    if bad:
        raise ValueError(f"CPU ids {bad} out of range 0-{n - 1}")
    return tuple(sorted(set(cpus)))


def format_cpu_list(cpus: tuple[int, ...]) -> str:
    runs, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + (None,):
        if c is not None and c == prev + 1:
            prev = c
            continue
        runs.append(f"{start}-{prev}" if prev != start else f"{start}")
        if c is not None:
            start = prev = c
    return ",".join(runs)


def apply_edge_hardware(hw: EdgeHardware) -> None:
    """Confine the *current* process to ``hw``. Call before loading models."""

    global _JOB_HANDLE
    count = hw.thread_count()
    if count is not None:
        os.environ["OMP_NUM_THREADS"] = str(count)
        import cv2
        import torch

        torch.set_num_threads(count)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # only settable before torch's first parallel work
        cv2.setNumThreads(count)

    if hw.cpu_cores is None and hw.cpus is None and hw.memory_mb is None:
        return
    if sys.platform != "win32":
        raise NotImplementedError("edge hardware limits use Windows Job Objects; on Linux use cgroups/taskset")

    import ctypes
    import ctypes.wintypes as wt

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD),
        ]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("Basic", BASIC), ("IoInfo", ctypes.c_uint64 * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class RATE(ctypes.Structure):
        _fields_ = [("ControlFlags", wt.DWORD), ("CpuRate", wt.DWORD)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = wt.HANDLE
    k.GetCurrentProcess.restype = wt.HANDLE

    def check(ok: int, what: str) -> None:
        if not ok:
            raise OSError(ctypes.get_last_error(), f"{what} failed")

    job = k.CreateJobObjectW(None, None)
    check(job, "CreateJobObject")

    limits = EXTENDED()
    limits.Basic.LimitFlags = 0x1000  # SILENT_BREAKAWAY_OK: children are not confined
    if hw.cpus:
        limits.Basic.LimitFlags |= 0x10  # AFFINITY
        limits.Basic.Affinity = sum(1 << c for c in hw.cpus)
    if hw.memory_mb is not None:
        limits.Basic.LimitFlags |= 0x100  # PROCESS_MEMORY
        limits.ProcessMemoryLimit = hw.memory_mb * 1024 * 1024
    check(k.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(limits), ctypes.sizeof(limits)),
          "SetInformationJobObject(limits)")

    if hw.cpu_cores is not None:
        # CpuRate is in 1/100ths of a percent of *all* logical CPUs.
        rate = max(1, min(10000, round(hw.cpu_cores / (os.cpu_count() or 1) * 10000)))
        info = RATE(0x1 | 0x4, rate)  # ENABLE | HARD_CAP
        check(k.SetInformationJobObject(wt.HANDLE(job), 15, ctypes.byref(info), ctypes.sizeof(info)),
              "SetInformationJobObject(cpu rate)")

    check(k.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(k.GetCurrentProcess())), "AssignProcessToJobObject")
    _JOB_HANDLE = job


class CpuMeter:
    """CPU actually consumed by this process, in cores' worth, between marks."""

    def __init__(self) -> None:
        import psutil

        self._proc = psutil.Process()
        self.reset()

    def reset(self) -> None:
        t = self._proc.cpu_times()
        self._cpu0, self._wall0 = t.user + t.system, time.perf_counter()

    def cores_used(self) -> float:
        t = self._proc.cpu_times()
        wall = time.perf_counter() - self._wall0
        return (t.user + t.system - self._cpu0) / wall if wall > 0 else 0.0
