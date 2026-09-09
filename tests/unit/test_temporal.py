"""Tests for `pragma.processing.temporal` (written in Phase 2, untested until Phase 5's
architecture test pass called out calendar-feature periodicity — section 16.4)."""

import math
from datetime import UTC, datetime

import pytest

from pragma.processing.temporal import (
    calendar_features,
    soft_log_time,
    time_to_evaluation,
    time_to_latest,
)


def test_soft_log_time_zero_is_zero() -> None:
    assert soft_log_time(0.0) == 0.0


def test_soft_log_time_is_monotonically_increasing() -> None:
    values = [soft_log_time(t) for t in [0, 60, 3600, 86400, 86400 * 30]]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_soft_log_time_rejects_negative_deltas() -> None:
    with pytest.raises(ValueError):
        soft_log_time(-1.0)


def test_calendar_features_hour_periodicity() -> None:
    """Hour 0 and hour 24 (next day, same hour) must produce identical sin/cos."""
    day1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    day2 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=UTC)
    f1, f2 = calendar_features(day1), calendar_features(day2)
    assert f1.hour_sin == pytest.approx(f2.hour_sin)
    assert f1.hour_cos == pytest.approx(f2.hour_cos)


def test_calendar_features_weekday_periodicity() -> None:
    """Two Mondays a week apart must produce identical weekday sin/cos."""
    monday1 = datetime(2026, 1, 5, 12, 0, 0, tzinfo=UTC)
    monday2 = datetime(2026, 1, 12, 12, 0, 0, tzinfo=UTC)
    assert monday1.weekday() == monday2.weekday() == 0
    f1, f2 = calendar_features(monday1), calendar_features(monday2)
    assert f1.dow_sin == pytest.approx(f2.dow_sin)
    assert f1.dow_cos == pytest.approx(f2.dow_cos)


def test_calendar_features_distinct_hours_differ() -> None:
    noon = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    midnight = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    f_noon, f_midnight = calendar_features(noon), calendar_features(midnight)
    assert (f_noon.hour_sin, f_noon.hour_cos) != (f_midnight.hour_sin, f_midnight.hour_cos)


def test_calendar_features_are_unit_norm_per_cycle() -> None:
    ts = datetime(2026, 3, 17, 9, 41, 0, tzinfo=UTC)
    f = calendar_features(ts)
    assert math.hypot(f.hour_sin, f.hour_cos) == pytest.approx(1.0)
    assert math.hypot(f.dow_sin, f.dow_cos) == pytest.approx(1.0)
    assert math.hypot(f.dom_sin, f.dom_cos) == pytest.approx(1.0)


def test_time_to_latest_matches_soft_log_of_elapsed_seconds() -> None:
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 1, 2, tzinfo=UTC)
    expected = soft_log_time(86400.0)
    assert time_to_latest(earlier, later) == pytest.approx(expected)


def test_time_to_evaluation_absent_milestone_is_zero() -> None:
    evaluation_time = datetime(2026, 1, 1, tzinfo=UTC)
    assert time_to_evaluation(None, evaluation_time) == 0.0


def test_time_to_evaluation_present_milestone_matches_elapsed_time() -> None:
    milestone = datetime(2026, 1, 1, tzinfo=UTC)
    evaluation_time = datetime(2026, 1, 8, tzinfo=UTC)
    expected = soft_log_time(7 * 86400.0)
    assert time_to_evaluation(milestone, evaluation_time) == pytest.approx(expected)
