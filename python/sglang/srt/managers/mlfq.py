from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Iterable, Optional, Sequence, Tuple


DEFAULT_MLFQ_QUANTA = (1, 2, 4)
DEFAULT_MLFQ_PREFILL_THRESHOLDS = (32, 256)
DEFAULT_MLFQ_DECODE_THRESHOLDS = (32, 256)
DEFAULT_MLFQ_STARVATION_SECONDS = 3.0
DEFAULT_MLFQ_ELASTIC_LONG_REQUEST_TOKENS = 256
DEFAULT_MLFQ_ELASTIC_MIN_COMPLETED_REQUESTS = 16
DEFAULT_MLFQ_ELASTIC_SERVICE_TIME_FLOOR_SECONDS = 1e-6
DEFAULT_MLFQ_DECODE_SECONDS_PER_TOKEN_EWMA_ALPHA = 0.1


@dataclasses.dataclass(frozen=True)
class MLFQConfig:
    """Configuration for MLFQ admission scheduling.

    This is not a preemptive OS-style MLFQ. It only orders requests while they
    are waiting to be admitted to prefill or preallocated decode work.
    """

    quanta: Tuple[int, ...] = DEFAULT_MLFQ_QUANTA
    prefill_thresholds: Tuple[int, ...] = DEFAULT_MLFQ_PREFILL_THRESHOLDS
    decode_thresholds: Tuple[int, ...] = DEFAULT_MLFQ_DECODE_THRESHOLDS
    starvation_seconds: float = DEFAULT_MLFQ_STARVATION_SECONDS
    elastic_slowdown_multiplier: Optional[float] = None
    elastic_long_request_tokens: int = DEFAULT_MLFQ_ELASTIC_LONG_REQUEST_TOKENS
    elastic_min_completed_requests: int = (
        DEFAULT_MLFQ_ELASTIC_MIN_COMPLETED_REQUESTS
    )
    elastic_service_time_floor_seconds: float = (
        DEFAULT_MLFQ_ELASTIC_SERVICE_TIME_FLOOR_SECONDS
    )

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
        if len(self.decode_thresholds) != len(self.quanta) - 1:
            raise ValueError(
                "--mlfq-decode-thresholds must contain exactly one fewer value "
                "than --mlfq-quanta"
            )
        if any(t <= 0 for t in self.prefill_thresholds):
            raise ValueError(
                "--mlfq-prefill-thresholds values must be positive integers"
            )
        if any(t <= 0 for t in self.decode_thresholds):
            raise ValueError(
                "--mlfq-decode-thresholds values must be positive integers"
            )
        if any(
            self.prefill_thresholds[i] >= self.prefill_thresholds[i + 1]
            for i in range(len(self.prefill_thresholds) - 1)
        ):
            raise ValueError("--mlfq-prefill-thresholds must be strictly increasing")
        if any(
            self.decode_thresholds[i] >= self.decode_thresholds[i + 1]
            for i in range(len(self.decode_thresholds) - 1)
        ):
            raise ValueError("--mlfq-decode-thresholds must be strictly increasing")
        if not math.isfinite(self.starvation_seconds):
            raise ValueError("--mlfq-starvation-seconds must be finite")
        if self.starvation_seconds <= 0:
            raise ValueError("--mlfq-starvation-seconds must be positive")
        if self.elastic_slowdown_multiplier is not None:
            if not math.isfinite(self.elastic_slowdown_multiplier):
                raise ValueError(
                    "--mlfq-elastic-slowdown-multiplier must be finite"
                )
            if self.elastic_slowdown_multiplier <= 1.0:
                raise ValueError(
                    "--mlfq-elastic-slowdown-multiplier must be > 1.0"
                )
        if self.elastic_long_request_tokens < 1:
            raise ValueError("--mlfq-elastic-long-request-tokens must be >= 1")
        if self.elastic_min_completed_requests < 0:
            raise ValueError("--mlfq-elastic-min-completed-requests must be >= 0")
        if not math.isfinite(self.elastic_service_time_floor_seconds):
            raise ValueError(
                "--mlfq-elastic-service-time-floor-seconds must be finite"
            )
        if self.elastic_service_time_floor_seconds <= 0:
            raise ValueError(
                "--mlfq-elastic-service-time-floor-seconds must be positive"
            )

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
            decode_thresholds=getattr(
                server_args,
                "mlfq_decode_thresholds",
                DEFAULT_MLFQ_DECODE_THRESHOLDS,
            ),
            starvation_seconds=getattr(
                server_args,
                "mlfq_starvation_seconds",
                DEFAULT_MLFQ_STARVATION_SECONDS,
            ),
            elastic_slowdown_multiplier=getattr(
                server_args,
                "mlfq_elastic_slowdown_multiplier",
                None,
            ),
            elastic_long_request_tokens=getattr(
                server_args,
                "mlfq_elastic_long_request_tokens",
                DEFAULT_MLFQ_ELASTIC_LONG_REQUEST_TOKENS,
            ),
            elastic_min_completed_requests=getattr(
                server_args,
                "mlfq_elastic_min_completed_requests",
                DEFAULT_MLFQ_ELASTIC_MIN_COMPLETED_REQUESTS,
            ),
            elastic_service_time_floor_seconds=getattr(
                server_args,
                "mlfq_elastic_service_time_floor_seconds",
                DEFAULT_MLFQ_ELASTIC_SERVICE_TIME_FLOOR_SECONDS,
            ),
        )

    @classmethod
    def from_cli_values(
        cls,
        *,
        quanta: str | Sequence[int],
        prefill_thresholds: str | Sequence[int],
        starvation_seconds: float,
        decode_thresholds: str | Sequence[int] = DEFAULT_MLFQ_DECODE_THRESHOLDS,
        elastic_slowdown_multiplier: Optional[float] = None,
        elastic_long_request_tokens: int = DEFAULT_MLFQ_ELASTIC_LONG_REQUEST_TOKENS,
        elastic_min_completed_requests: int = (
            DEFAULT_MLFQ_ELASTIC_MIN_COMPLETED_REQUESTS
        ),
        elastic_service_time_floor_seconds: float = (
            DEFAULT_MLFQ_ELASTIC_SERVICE_TIME_FLOOR_SECONDS
        ),
    ) -> "MLFQConfig":
        return cls(
            quanta=_parse_positive_int_csv("--mlfq-quanta", quanta),
            prefill_thresholds=_parse_positive_int_csv(
                "--mlfq-prefill-thresholds", prefill_thresholds
            ),
            decode_thresholds=_parse_positive_int_csv(
                "--mlfq-decode-thresholds", decode_thresholds
            ),
            starvation_seconds=float(starvation_seconds),
            elastic_slowdown_multiplier=(
                None
                if elastic_slowdown_multiplier is None
                else float(elastic_slowdown_multiplier)
            ),
            elastic_long_request_tokens=int(elastic_long_request_tokens),
            elastic_min_completed_requests=int(elastic_min_completed_requests),
            elastic_service_time_floor_seconds=float(
                elastic_service_time_floor_seconds
            ),
        )

    def level_for_prefill_work(self, work_tokens: int) -> int:
        return _level_for_work(work_tokens, self.prefill_thresholds, self.num_levels)

    def level_for_decode_work(self, work_tokens: int) -> int:
        return _level_for_work(work_tokens, self.decode_thresholds, self.num_levels)

    def remaining_decode_tokens(self, req: Any) -> int:
        sampling_params = getattr(req, "sampling_params", None)
        max_new_tokens = getattr(sampling_params, "max_new_tokens", 1)
        return max(1, int(max_new_tokens) - len(getattr(req, "output_ids", ())))

    def elastic_enabled(self) -> bool:
        return self.elastic_slowdown_multiplier is not None

    def elastic_effective_threshold(self, stats: "DecodeMLFQStats") -> float:
        if not self.elastic_enabled():
            return 0.0
        return (
            max(1.0, stats.completed_slowdown_mean)
            * self.elastic_slowdown_multiplier
        )

    def service_time_floor(self) -> float:
        return self.elastic_service_time_floor_seconds

    def next_level_after_service(
        self, level: int, tokens_in_level: int
    ) -> tuple[int, int]:
        level = min(max(level, 0), self.num_levels - 1)
        quantum = self.quanta[level]
        if tokens_in_level >= quantum:
            return min(level + 1, self.num_levels - 1), 0
        return level, tokens_in_level


@dataclasses.dataclass
class DecodeMLFQStats:
    completed_count: int = 0
    completed_slowdown_sum: float = 0.0
    decode_seconds_per_token_ewma: Optional[float] = None
    ewma_alpha: float = DEFAULT_MLFQ_DECODE_SECONDS_PER_TOKEN_EWMA_ALPHA

    @property
    def completed_slowdown_mean(self) -> float:
        if self.completed_count == 0:
            return 0.0
        return self.completed_slowdown_sum / self.completed_count

    def record_completed_request(
        self,
        *,
        d_flow_time: float,
        d_service_time: float,
        output_tokens: int,
        epsilon: float,
    ) -> float:
        service_time = max(float(d_service_time), epsilon)
        slowdown = max(0.0, float(d_flow_time)) / service_time
        self.completed_count += 1
        self.completed_slowdown_sum += slowdown

        tokens = max(1, int(output_tokens))
        seconds_per_token = service_time / tokens
        if self.decode_seconds_per_token_ewma is None:
            self.decode_seconds_per_token_ewma = seconds_per_token
        else:
            alpha = min(max(float(self.ewma_alpha), 0.0), 1.0)
            self.decode_seconds_per_token_ewma = (
                alpha * seconds_per_token
                + (1.0 - alpha) * self.decode_seconds_per_token_ewma
            )
        return slowdown


def _level_for_work(work_tokens: int, thresholds: Tuple[int, ...], num_levels: int) -> int:
    work_tokens = max(1, int(work_tokens))
    for level, threshold in enumerate(thresholds):
        if work_tokens <= threshold:
            return level
    return num_levels - 1


def assign_initial_decode_mlfq_level(req: Any, config: MLFQConfig) -> None:
    req.mlfq_level = config.level_for_decode_work(config.remaining_decode_tokens(req))
    req.mlfq_tokens_in_level = 0
    req.mlfq_classified_for_queue = True
    req.mlfq_decode_initialized = True


def predict_decode_slowdown(
    req: Any,
    *,
    now: float,
    d_queue_entry_time: float,
    stats: DecodeMLFQStats,
    config: MLFQConfig,
) -> tuple[float, float, float]:
    remaining = config.remaining_decode_tokens(req)
    seconds_per_token = stats.decode_seconds_per_token_ewma
    if seconds_per_token is None:
        seconds_per_token = config.service_time_floor()
    observed_service_time = float(
        getattr(req, "decode_batch_wall_time_attributed", 0.0)
        or getattr(req, "decode_execution_time", 0.0)
        or 0.0
    )
    predicted_remaining_service_time = remaining * max(
        seconds_per_token, config.service_time_floor()
    )
    predicted_service_time = observed_service_time + predicted_remaining_service_time
    predicted_flow_time = max(0.0, now - d_queue_entry_time) + (
        predicted_service_time - observed_service_time
    )
    predicted_slowdown = predicted_flow_time / max(
        predicted_service_time, config.service_time_floor()
    )
    return predicted_slowdown, predicted_service_time, predicted_flow_time


def maybe_elastic_promote(
    req: Any,
    *,
    now: float,
    d_queue_entry_time: float,
    stats: DecodeMLFQStats,
    config: MLFQConfig,
) -> bool:
    threshold = config.elastic_effective_threshold(stats)
    req.elastic_effective_threshold = threshold

    if not config.elastic_enabled():
        req.predicted_slowdown_at_last_queue_check = 0.0
        return False
    if stats.completed_count < config.elastic_min_completed_requests:
        req.predicted_slowdown_at_last_queue_check = 0.0
        return False

    remaining = config.remaining_decode_tokens(req)
    if remaining < config.elastic_long_request_tokens:
        req.predicted_slowdown_at_last_queue_check = 0.0
        return False

    predicted_slowdown, _, _ = predict_decode_slowdown(
        req,
        now=now,
        d_queue_entry_time=d_queue_entry_time,
        stats=stats,
        config=config,
    )
    req.predicted_slowdown_at_last_queue_check = predicted_slowdown
    if predicted_slowdown < threshold:
        return False
    if getattr(req, "mlfq_elastic_promoted", False):
        return False

    req.mlfq_level = 0
    req.mlfq_tokens_in_level = 0
    req.mlfq_queue_enter_time = now
    req.mlfq_classified_for_queue = True
    req.mlfq_elastic_promoted = True
    return True


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
