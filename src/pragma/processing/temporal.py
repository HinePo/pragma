"""Temporal feature computation (implementation plan, section 6.1, 7.3).

Produces the raw scalar/vector features later consumed by `ContinuousRoPE`
and `CalendarEncoder` (Phase 5). This module is stateless/unfitted: the
soft-log transform and cyclical calendar conversion use fixed formulas, not
data-dependent parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


def soft_log_time(delta_seconds: float, *, scale: float = 8.0) -> float:
    """Soft-log time transform: `f(t) = scale * ln(1 + t / scale)`.

    `delta_seconds` must be >= 0 (elapsed time, never negative under
    point-in-time correctness). Implementation plan section 7.3.
    """
    if delta_seconds < 0:
        raise ValueError("delta_seconds must be non-negative under point-in-time correctness")
    return scale * math.log1p(delta_seconds / scale)


@dataclass(frozen=True)
class CalendarFeatures:
    """Fixed sine/cosine cyclical encoding of hour, weekday, and day-of-month."""

    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    dom_sin: float
    dom_cos: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.dom_sin,
            self.dom_cos,
        )


def calendar_features(timestamp: datetime) -> CalendarFeatures:
    hour_angle = 2 * math.pi * (timestamp.hour / 24.0)
    dow_angle = 2 * math.pi * (timestamp.weekday() / 7.0)
    # Days-in-month varies; day-of-month is cycled over a fixed 31-day period,
    # which is an approximation but keeps the encoding parameter-free.
    dom_angle = 2 * math.pi * ((timestamp.day - 1) / 31.0)
    return CalendarFeatures(
        hour_sin=math.sin(hour_angle),
        hour_cos=math.cos(hour_angle),
        dow_sin=math.sin(dow_angle),
        dow_cos=math.cos(dow_angle),
        dom_sin=math.sin(dom_angle),
        dom_cos=math.cos(dom_angle),
    )


def time_to_latest(event_time: datetime, latest_time: datetime) -> float:
    """History temporal coordinate: soft-log elapsed time from `event_time` to `latest_time`."""
    return soft_log_time((latest_time - event_time).total_seconds())


def time_to_evaluation(milestone_time: datetime | None, evaluation_time: datetime) -> float:
    """Profile temporal coordinate: soft-log elapsed time from a milestone to evaluation.

    Static (non-milestone) profile attributes receive `0.0` directly (section
    7.3); this function is only for milestone fields, and returns `0.0` for
    an absent milestone as well, since `[MILESTONE_ABSENT]` already encodes
    non-occurrence at the token level.
    """
    if milestone_time is None:
        return 0.0
    return soft_log_time((evaluation_time - milestone_time).total_seconds())
