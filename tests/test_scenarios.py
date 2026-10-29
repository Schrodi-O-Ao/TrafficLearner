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


def test_map_direction_string_bremen():
    """Verify Bremen direction numbers and SUMO routes map correctly to 4 approaches."""
    from backend.data_ingest import _map_direction_string
    assert _map_direction_string("Direction 1") == "N"
    assert _map_direction_string("Direction 6") == "W"
    assert _map_direction_string("Direction 8") == "S"
    assert _map_direction_string("Direction 13") == "E"
    assert _map_direction_string("route_12") == "E"
    assert _map_direction_string("route_0") == "N"


def test_load_from_bremen_ods_format_csv():
    """Verify CSV with original CN+ Bremen columns Date, Time, Direction, Number."""
    from backend.data_ingest import load_traffic_dataset
    csv_content = (
        "Date,Time,Direction,Number,Type,Temperature\n"
        "2022.05.24,17:00:01,Direction 13,2,Normal,61 °F\n"
        "2022.05.24,17:00:24,Direction 6,3,Normal,61 °F\n"
        "2022.05.24,17:01:05,Direction 1,1,Normal,61 °F\n"
        "2022.05.24,17:01:30,Direction 8,1,Normal,61 °F\n"
    )
    rows = load_traffic_dataset(csv_content.encode("utf-8"), filename="CN+ Dataset.csv")
    assert len(rows) == 48
    directions = {r["direction"] for r in rows}
    assert directions == {"N", "S", "E", "W"}


def test_load_from_sumo_xml():
    """Verify SUMO .rou.xml route files are parsed directly."""
    from backend.data_ingest import load_traffic_dataset
    xml_content = (
        "<routes>\n"
        "  <vehicle id=\"v0\" depart=\"1.0\" route=\"route_0\"/>\n"
        "  <vehicle id=\"v1\" depart=\"2.5\" route=\"route_12\"/>\n"
        "</routes>"
    )
    rows = load_traffic_dataset(xml_content.encode("utf-8"), filename="scenario.rou.xml")
    assert len(rows) == 48


def test_load_from_zip():
    """Verify ZIP archives containing traffic datasets are extracted and parsed."""
    import zipfile
    import io
    from backend.data_ingest import load_traffic_dataset

    csv_data = b"time_seconds,direction,arrival_rate\n0,N,0.1\n0,S,0.1\n"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("traffic.csv", csv_data)

    rows = load_traffic_dataset(zip_buffer.getvalue(), filename="dataset.zip")
    assert len(rows) == 2
