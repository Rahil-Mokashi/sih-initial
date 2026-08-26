"""Tests for src/attribution/score_vessels.py's scoring math."""

from datetime import datetime, timezone

from attribution.score_vessels import score_vessels, time_gap_hours


def test_time_gap_zero_when_origin_inside_presence_window():
    origin = datetime(2019, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    entry = datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    exit_ = datetime(2019, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert time_gap_hours(origin, entry, exit_) == 0.0


def test_time_gap_measured_to_nearest_edge_before_window():
    origin = datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    entry = datetime(2019, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    exit_ = datetime(2019, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert time_gap_hours(origin, entry, exit_) == 5.0


def test_time_gap_measured_to_nearest_edge_after_window():
    origin = datetime(2019, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
    entry = datetime(2019, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    exit_ = datetime(2019, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert time_gap_hours(origin, entry, exit_) == 5.0


def _record(vessel_id, lon, lat, entry, exit_, **extra):
    return {
        "vesselId": vessel_id, "lon": lon, "lat": lat,
        "entryTimestamp": entry, "exitTimestamp": exit_,
        "mmsi": extra.get("mmsi"), "imo": extra.get("imo"),
        "shipName": extra.get("shipName"), "flag": extra.get("flag"),
        "vesselType": extra.get("vesselType"), "date": extra.get("date", "2019-01-01"),
        "hours": extra.get("hours", 1),
    }


def test_closer_vessel_ranks_first():
    origin = (33.0, 33.0)
    origin_time = datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        _record("far", 34.0, 34.0, "2019-01-01T12:00:00Z", "2019-01-01T13:00:00Z", shipName="FAR"),
        _record("near", 33.01, 33.01, "2019-01-01T12:00:00Z", "2019-01-01T13:00:00Z", shipName="NEAR"),
    ]
    ranked = score_vessels(origin, origin_time, records)
    assert ranked[0].vessel_id == "near"
    assert ranked[0].score < ranked[1].score


def test_missing_vessel_id_is_skipped():
    origin = (33.0, 33.0)
    origin_time = datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [{"lon": 33.0, "lat": 33.0, "entryTimestamp": "2019-01-01T12:00:00Z", "exitTimestamp": "2019-01-01T13:00:00Z"}]
    ranked = score_vessels(origin, origin_time, records)
    assert ranked == []


def test_vessel_with_multiple_rows_keeps_only_its_best_score():
    origin = (33.0, 33.0)
    origin_time = datetime(2019, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        _record("v1", 40.0, 40.0, "2019-01-01T12:00:00Z", "2019-01-01T13:00:00Z"),  # far row
        _record("v1", 33.0, 33.0, "2019-01-01T12:00:00Z", "2019-01-01T13:00:00Z"),  # close row, same vessel
    ]
    ranked = score_vessels(origin, origin_time, records)
    assert len(ranked) == 1
    assert ranked[0].distance_km < 1.0  # kept the close row, not the far one
