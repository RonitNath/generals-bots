from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import jax


def _block_ready(value: Any) -> Any:
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


@dataclass
class TimerStat:
    count: int = 0
    total: float = 0.0
    min: float = field(default_factory=lambda: float("inf"))
    max: float = 0.0

    def add(self, duration: float) -> None:
        self.count += 1
        self.total += duration
        self.min = min(self.min, duration)
        self.max = max(self.max, duration)

    def to_dict(self) -> dict[str, float | int]:
        avg = self.total / self.count if self.count else 0.0
        min_value = 0.0 if self.min == float("inf") else self.min
        return {
            "count": self.count,
            "total_sec": self.total,
            "avg_sec": avg,
            "min_sec": min_value,
            "max_sec": self.max,
        }


class Telemetry:
    def __init__(self):
        self._stats: dict[str, TimerStat] = defaultdict(TimerStat)
        self._samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def record(self, name: str, duration: float) -> None:
        self._stats[name].add(duration)

    def time_block(self, name: str, fn, *, ready_value: Any = None):
        start = time.perf_counter()
        result = fn()
        _block_ready(ready_value if ready_value is not None else result)
        self.record(name, time.perf_counter() - start)
        return result

    def add_sample(self, name: str, payload: dict[str, Any], *, limit: int = 10) -> None:
        bucket = self._samples[name]
        bucket.append(payload)
        if len(bucket) > limit:
            del bucket[0]

    def merge(self, other: "Telemetry", prefix: str | None = None) -> None:
        for name, stat in other._stats.items():
            target_name = f"{prefix}.{name}" if prefix else name
            target = self._stats[target_name]
            target.count += stat.count
            target.total += stat.total
            target.min = min(target.min, stat.min)
            target.max = max(target.max, stat.max)
        for name, samples in other._samples.items():
            target_name = f"{prefix}.{name}" if prefix else name
            self._samples[target_name].extend(samples)

    def snapshot(self) -> dict[str, Any]:
        return {
            "timings": {name: stat.to_dict() for name, stat in sorted(self._stats.items())},
            "samples": {name: list(samples) for name, samples in sorted(self._samples.items())},
        }
