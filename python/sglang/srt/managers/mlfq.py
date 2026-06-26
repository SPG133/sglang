from __future__ import annotations

import dataclasses
import time
from typing import Any, Iterable, Sequence, Tuple


DEFAULT_MLFQ_QUANTA = (1, 2, 4)
DEFAULT_MLFQ_PREFILL_THRESHOLDS = (32, 256)
DEFAULT_MLFQ_STARVATION_SECONDS = 1.0


@dataclasses.dataclass(frozen=True)
class MLFQConfig:
    """Configuration for MLFQ admission scheduling.

    This is not a preemptive OS-style MLFQ. It only orders requests while they
    are waiting to be admitted to prefill or preallocated decode work.
    """

    quanta: Tuple[int, ...] = DEFAULT_MLFQ_QUANTA
    prefill_thresholds: Tuple[int, ...] = DEFAULT_MLFQ_PREFILL_THRESHOLDS
    starvation_seconds: float = DEFAULT_MLFQ_STARVATION_SECONDS

    def __post_init__(self):
        if len(self.quanta) == 0:
            raise ValueError("--mlfq-quanta must contain at least one integer")
        if any(q <= 0 for q in self.quanta):
            raise ValueError("--mlfq-quanta values must be positive integers")
        if len(self.prefill_thresholds) != len(self.quanta) - 1:
            raise ValueError(
                "--mlfq-prefill-thresholds must contain exactly one fewer value "
                "than --mlfq-quanta"
            )
        if any(t <= 0 for t in self.prefill_thresholds):
            raise ValueError(
                "--mlfq-prefill-thresholds values must be positive integers"
            )
        if any(
            self.prefill_thresholds[i] >= self.prefill_thresholds[i + 1]
            for i in range(len(self.prefill_thresholds) - 1)
        ):
            raise ValueError("--mlfq-prefill-thresholds must be strictly increasing")
        if self.starvation_seconds <= 0:
            raise ValueError("--mlfq-starvation-seconds must be positive")

    @property
    def num_levels(self) -> int:
        return len(self.quanta)

    @classmethod
    def from_server_args(cls, server_args: Any) -> "MLFQConfig":
        return cls.from_cli_values(
            quanta=getattr(server_args, "mlfq_quanta", DEFAULT_MLFQ_QUANTA),
            prefill_thresholds=getattr(
                server_args,
                "mlfq_prefill_thresholds",
                DEFAULT_MLFQ_PREFILL_THRESHOLDS,
            ),
            starvation_seconds=getattr(
                server_args,
                "mlfq_starvation_seconds",
                DEFAULT_MLFQ_STARVATION_SECONDS,
            ),
        )

    @classmethod
    def from_cli_values(
        cls,
        *,
        quanta: str | Sequence[int],
        prefill_thresholds: str | Sequence[int],
        starvation_seconds: float,
    ) -> "MLFQConfig":
        return cls(
            quanta=_parse_positive_int_csv("--mlfq-quanta", quanta),
            prefill_thresholds=_parse_positive_int_csv(
                "--mlfq-prefill-thresholds", prefill_thresholds
            ),
            starvation_seconds=float(starvation_seconds),
        )

    def level_for_prefill_work(self, work_tokens: int) -> int:
        work_tokens = max(1, int(work_tokens))
        for level, threshold in enumerate(self.prefill_thresholds):
            if work_tokens <= threshold:
                return level
        return self.num_levels - 1

    def next_level_after_service(
        self, level: int, tokens_in_level: int
    ) -> tuple[int, int]:
        level = min(max(level, 0), self.num_levels - 1)
        quantum = self.quanta[level]
        if tokens_in_level >= quantum:
            return min(level + 1, self.num_levels - 1), 0
        return level, tokens_in_level


def _parse_positive_int_csv(name: str, value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if any(part == "" for part in parts):
            raise ValueError(f"{name} must be a comma-separated list of integers")
        try:
            parsed = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of integers"
            ) from exc
    else:
        parsed = tuple(int(part) for part in value)
    if len(parsed) == 0 or any(part <= 0 for part in parsed):
        raise ValueError(f"{name} values must be positive integers")
    return parsed


def assign_initial_mlfq_level(req: Any, work_tokens: int, config: MLFQConfig) -> None:
    req.mlfq_level = config.level_for_prefill_work(work_tokens)
    req.mlfq_tokens_in_level = 0
    req.mlfq_classified_for_queue = True


def update_mlfq_after_service(
    req: Any, scheduled_tokens: int, config: MLFQConfig
) -> None:
    req.mlfq_tokens_in_level += max(1, int(scheduled_tokens))
    req.mlfq_level, req.mlfq_tokens_in_level = config.next_level_after_service(
        getattr(req, "mlfq_level", config.num_levels - 1),
        getattr(req, "mlfq_tokens_in_level", 0),
    )


def record_mlfq_enqueue(req: Any, timestamp: float | None = None) -> None:
    req.mlfq_queue_enter_time = (
        timestamp if timestamp is not None else time.monotonic()
    )
    req.mlfq_is_queued = True
    req.mlfq_classified_for_queue = False


def record_mlfq_dequeue(req: Any, timestamp: float | None = None) -> None:
    if not getattr(req, "mlfq_is_queued", False):
        return
    now = timestamp if timestamp is not None else time.monotonic()
    enter_time = getattr(req, "mlfq_queue_enter_time", 0.0)
    req.mlfq_last_wait_duration = max(0.0, now - enter_time)
    req.mlfq_is_queued = False


def promote_starved_requests(
    reqs: Iterable[Any], config: MLFQConfig, timestamp: float | None = None
) -> None:
    now = timestamp if timestamp is not None else time.monotonic()
    for req in reqs:
        if not getattr(req, "mlfq_is_queued", False):
            continue
        wait_duration = max(0.0, now - getattr(req, "mlfq_queue_enter_time", now))
        req.mlfq_last_wait_duration = wait_duration
        if wait_duration >= config.starvation_seconds:
            req.mlfq_level = 0
            req.mlfq_tokens_in_level = 0
            req.mlfq_queue_enter_time = now
            req.mlfq_classified_for_queue = True


def mlfq_sort_key(obj: Any, config: MLFQConfig) -> tuple[int, float, float, str]:
    req = getattr(obj, "req", obj)
    fallback_level = config.num_levels - 1
    time_stats = getattr(req, "time_stats", None)
    return (
        int(getattr(req, "mlfq_level", fallback_level)),
        float(getattr(req, "mlfq_queue_enter_time", float("inf"))),
        float(getattr(time_stats, "wait_queue_entry_time", 0.0)),
        str(getattr(req, "rid", "")),
    )
