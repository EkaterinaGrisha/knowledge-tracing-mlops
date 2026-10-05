"""Infrastructure monitoring: wall-clock time, peak RSS memory, CPU utilisation.

Used as a context manager around training stages; the captured metrics are
logged to MLflow so model-quality and infrastructure cost can be tracked
side by side across runs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceStats:
    label: str
    elapsed_s: float = 0.0
    peak_rss_mb: float = 0.0
    avg_cpu_pct: float = 0.0
    samples: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "elapsed_s": round(self.elapsed_s, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "avg_cpu_pct": round(self.avg_cpu_pct, 1),
        }


class ResourceMonitor:
    """Background sampler of process RSS + CPU%. Use as a context manager."""

    def __init__(self, label: str, interval: float = 0.5):
        self.stats = ResourceStats(label=label)
        self._interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _sample(self) -> None:
        self._proc.cpu_percent(None)  # prime the CPU% counter
        while not self._stop.is_set():
            rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            self.stats.peak_rss_mb = max(self.stats.peak_rss_mb, rss_mb)
            self.stats.samples.append(self._proc.cpu_percent(None))
            self._stop.wait(self._interval)

    def __enter__(self) -> ResourceMonitor:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.stats.elapsed_s = time.perf_counter() - self._t0
        if self.stats.samples:
            self.stats.avg_cpu_pct = sum(self.stats.samples) / len(self.stats.samples)
