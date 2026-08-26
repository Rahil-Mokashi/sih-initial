"""
Tests for src/drift/currents.py -- especially cycle_and_tau, which had a
real bug in this project: rounding a target time within ~30s of the next
cycle's 12Z could round tau up to exactly 24, one past the valid 0-23
range, requesting a HYCOM file that doesn't exist (see DECISIONS.md
"Drift model approach"). These tests exist so that bug can't come back
silently.
"""

from datetime import datetime, timezone

from drift.currents import cycle_and_tau


def test_exact_cycle_time_is_tau_zero():
    t = datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    cycle, tau = cycle_and_tau(t)
    assert cycle == datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert tau == 0


def test_before_noon_uses_previous_days_cycle():
    t = datetime(2019, 1, 1, 3, 42, 35, tzinfo=timezone.utc)
    cycle, tau = cycle_and_tau(t)
    assert cycle == datetime(2018, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
    assert tau == 16  # 15h42m35s rounds to 16


def test_after_noon_uses_same_days_cycle():
    t = datetime(2019, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
    cycle, tau = cycle_and_tau(t)
    assert cycle == datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert tau == 3


def test_rounding_to_tau_24_rolls_over_to_next_cycle():
    """
    The real bug: a target time 23h59m31s+ past a cycle's 12Z rounds tau
    up to 24 -- one past that cycle's valid 0-23 range. Must roll over to
    the NEXT cycle's tau=0 instead of requesting a nonexistent tau=24 file.
    """
    # 2018-12-31 11:59:31 is 23h59m31s after the 2018-12-30 12:00 cycle --
    # this specific second was the one that triggered the real bug.
    t = datetime(2018, 12, 31, 11, 59, 31, tzinfo=timezone.utc)
    cycle, tau = cycle_and_tau(t)
    assert tau < 24, "tau must never be 24 or more -- that file doesn't exist"
    assert cycle == datetime(2018, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
    assert tau == 0


def test_tau_always_in_valid_range_across_a_full_day():
    """Sweep every hour of a day and confirm tau always lands in [0, 23]."""
    for hour in range(24):
        for minute in (0, 29, 59):
            t = datetime(2019, 1, 5, hour, minute, 0, tzinfo=timezone.utc)
            _, tau = cycle_and_tau(t)
            assert 0 <= tau <= 23, f"tau={tau} out of range for {t.isoformat()}"
