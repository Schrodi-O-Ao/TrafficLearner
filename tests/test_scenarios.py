"""Unit tests for scenario profiles, CSV loading, and arrival generation."""

import tempfile
import pytest
from backend.data_ingest import generate_synthetic_dataset, load_from_csv
from scenario_profiles import PRESET_PROFILES, profile_to_dataset, sample_arrival_schedule


def test_preset_profiles_exist():
    """Verify standard and T-junction preset profiles are defined with valid rates."""
    assert "balanced" in PRESET_PROFILES
    assert "heavy" in PRESET_PROFILES
    assert "t_junction" in PRESET_PROFILES

    for name, profile in PRESET_PROFILES.items():
        assert len(profile) >= 3
        for direction, segments in profile.items():
            assert direction in ("N", "S", "E", "W")
            assert len(segments) >= 1
            for start, rate in segments:
                assert start >= 0
                assert 0.0 <= rate <= 0.5


def test_profile_to_dataset():
    """Verify profile converts into sorted dataset rows."""
    dataset = profile_to_dataset(PRESET_PROFILES["balanced"])
    assert len(dataset) == 4
    for row in dataset:
        assert "time" in row
        assert "direction" in row
        assert "rate" in row


def test_sample_arrival_schedule():
    """Verify Poisson arrivals are sorted and non-negative."""
    dataset = profile_to_dataset(PRESET_PROFILES["balanced"])
    arrivals = sample_arrival_schedule(dataset, seed=42, duration=300)
    for direction in ("N", "S", "E", "W"):
        times = arrivals.get(direction, [])
        assert times == sorted(times)
        if times:
            assert min(times) >= 0.0
            assert max(times) <= 300.0


def test_load_from_csv():
    """Verify CSV loader parses time, direction, and arrival rate correctly."""
    csv_content = "time_seconds,direction,arrival_rate\n0,N,0.12\n0,S,0.10\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(csv_content)
        path = f.name

    try:
        rows = load_from_csv(path)
        assert len(rows) == 2
        assert rows[0]["direction"] in ("N", "S")
    finally:
        import os
        if os.path.exists(path):
            os.remove(path)


def test_load_from_bremen_csv():
    """Verify CSV loader recognizes real-world dataset formats like CN+ Bremen."""
    csv_content = (
        "Data Collection Date,Crossing Timing,Vehicle Direction,Number of Vehicles,Vehicle Type\n"
        "2022-05-24,08:00:05.120,North-South,1,Car\n"
        "2022-05-24,08:00:12.450,West-East,1,Car\n"
        "2022-05-24,08:01:22.000,South-North,1,Van\n"
        "2022-05-24,08:02:15.800,East-West,1,Car\n"
    )
    rows = load_from_csv(csv_content.encode("utf-8"))
    assert len(rows) == 48
    directions = {r["direction"] for r in rows}
    assert directions == {"N", "S", "E", "W"}
