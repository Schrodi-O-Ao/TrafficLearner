"""Validate uploaded traffic rate rows and sample reproducible vehicle arrival schedules."""

from __future__ import annotations

import csv
import io
import math
import re

import numpy as np

MAX_DATASET_ROWS = 50000
MAX_SIMULATION_SECONDS = 3600.0
MAX_ARRIVAL_RATE = 0.5
DIRECTIONS = ("N", "S", "E", "W")


def generate_synthetic_dataset(seed: int = 42) -> list[dict]:
    """Create seeded five-minute demand rows with a morning north-south peak."""
    rng = np.random.default_rng(seed)
    dataset = []
    for start_time in range(0, 3600, 300):
        morning_peak = start_time < 1800
        for direction in DIRECTIONS:
            if morning_peak and direction in ("N", "S"):
                rate = float(rng.uniform(0.08, 0.12))
            else:
                rate = float(rng.uniform(0.04, 0.07))
            dataset.append({"time": float(start_time), "direction": direction, "rate": round(rate, 4)})
    return dataset


def _parse_time_string(val: str) -> float | None:
    """Parse float seconds, HH:MM:SS, or HH:MM:SS.mmm into total seconds."""
    val = val.strip()
    if not val:
        return None
    time_match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?", val)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        seconds = int(time_match.group(3))
        frac = float("0." + time_match.group(4)) if time_match.group(4) else 0.0
        return hours * 3600.0 + minutes * 60.0 + seconds + frac
    try:
        return float(val)
    except ValueError:
        return None


def _map_direction_string(dir_str: str) -> str | None:
    """Map flexible direction names (North, NB, N-S, N, 1, etc.) to N, S, E, or W."""
    cleaned = str(dir_str).strip().upper()
    if not cleaned:
        return None
    # Direct match
    if cleaned in DIRECTIONS:
        return cleaned
    # Common prefixes and turning movements
    if cleaned.startswith("N") or cleaned in ("1", "NORTH", "NORTHBOUND", "NORTH-SOUTH", "N-S", "N->S"):
        return "N"
    if cleaned.startswith("S") or cleaned in ("2", "SOUTH", "SOUTHBOUND", "SOUTH-NORTH", "S-N", "S->N"):
        return "S"
    if cleaned.startswith("E") or cleaned in ("3", "EAST", "EASTBOUND", "EAST-WEST", "E-W", "E->W"):
        return "E"
    if cleaned.startswith("W") or cleaned in ("4", "WEST", "WESTBOUND", "WEST-EAST", "W-E", "W->E"):
        return "W"
    return None


def load_from_csv(filepath_or_bytes) -> list[dict]:
    """Read a CSV file supporting standard TrafficLearner schedules or datasets like CN+ Bremen."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                text_content = filepath_or_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV encoding not supported. Please save the file as UTF-8.")
        source = io.StringIO(text_content)
        close_source = False
    else:
        source = open(filepath_or_bytes, "r", encoding="utf-8-sig", newline="")
        close_source = True

    try:
        sample_text = source.read(4096)
        source.seek(0)

        # Detect delimiter: comma, semicolon, tab, or default comma
        delimiter = ","
        for delim in (",", ";", "\t"):
            if delim in sample_text and sample_text.count(delim) > sample_text.count(delimiter):
                delimiter = delim

        reader = csv.DictReader(source, delimiter=delimiter)
        raw_headers = reader.fieldnames or []
        normalized_headers = {h.strip().lower(): h for h in raw_headers}

        # Match Time column
        time_col = None
        for cand in (
            "time_seconds", "time", "timestamp", "crossing timing", "crossing_timing",
            "crossing_time", "crossing timing (s)", "time(s)", "seconds", "datetime"
        ):
            if cand in normalized_headers:
                time_col = normalized_headers[cand]
                break

        # Match Direction column
        dir_col = None
        for cand in (
            "direction", "vehicle direction", "vehicle_direction", "dir", "approach",
            "movement", "leg", "arm"
        ):
            if cand in normalized_headers:
                dir_col = normalized_headers[cand]
                break

        # Match Rate / Count column
        rate_col = None
        for cand in (
            "arrival_rate", "rate", "vehicle_count", "vehicle count", "number of vehicles",
            "number_of_vehicles", "count", "volume", "vehicles", "flow"
        ):
            if cand in normalized_headers:
                rate_col = normalized_headers[cand]
                break

        if not time_col or not dir_col:
            found = ", ".join(repr(h) for h in raw_headers)
            raise ValueError(
                f"CSV headers must include a time column and direction column. Found: [{found}]. "
                "Accepted time columns: time_seconds, time, crossing timing, timestamp. "
                "Accepted direction columns: direction, vehicle direction, approach."
            )

        raw_records = []
        for row_number, original_row in enumerate(reader, start=2):
            if row_number > MAX_DATASET_ROWS + 1:
                break

            time_raw = original_row.get(time_col)
            dir_raw = original_row.get(dir_col)
            if time_raw is None or dir_raw is None:
                continue

            t_val = _parse_time_string(str(time_raw))
            if t_val is None:
                continue

            d_val = _map_direction_string(str(dir_raw))
            if d_val is None:
                continue

            # Check rate or count
            rate_val = None
            if rate_col and original_row.get(rate_col) not in (None, ""):
                try:
                    parsed_val = float(original_row[rate_col])
                    # If the column name implies rate and is <= 1.0, treat as arrival rate
                    if "rate" in rate_col.lower() and parsed_val <= 1.0:
                        rate_val = parsed_val
                    else:
                        rate_val = parsed_val  # will be aggregated as count
                except ValueError:
                    rate_val = 1.0
            else:
                rate_val = 1.0  # 1 passage event

            raw_records.append((t_val, d_val, rate_val))

    finally:
        if close_source:
            source.close()

    if not raw_records:
        raise ValueError("CSV contains no recognizable traffic entries.")

    # Determine if this is an already-binned schedule (like sample_traffic.csv with explicit arrival_rate <= 0.5)
    # or raw individual vehicle passages (like CN+ Bremen dataset)
    is_prebinned_rate = (
        len(raw_records) <= 120
        and rate_col
        and "rate" in rate_col.lower()
        and all(0 <= r <= MAX_ARRIVAL_RATE for _, _, r in raw_records)
        and all(0 <= t <= MAX_SIMULATION_SECONDS for t, _, _ in raw_records)
    )

    if is_prebinned_rate:
        dataset = [{"time": float(t), "direction": d, "rate": float(r)} for t, d, r in raw_records]
        return sorted(dataset, key=lambda row: (row["direction"], row["time"]))

    # Aggregate raw vehicle passages into standard 300-second windows across 3600 seconds
    min_time = min(t for t, _, _ in raw_records)
    # Group by (window_index, direction)
    window_counts: dict[tuple[int, str], float] = {}
    for t, d, count in raw_records:
        rel_time = t - min_time
        if rel_time < 0 or rel_time >= MAX_SIMULATION_SECONDS:
            continue
        window_idx = int(rel_time // 300.0)
        key = (window_idx, d)
        window_counts[key] = window_counts.get(key, 0.0) + (count if count > 0 else 1.0)

    dataset = []
    for window_idx in range(12):  # 12 x 300s = 3600s
        time_sec = float(window_idx * 300)
        for direction in DIRECTIONS:
            count = window_counts.get((window_idx, direction), 0.0)
            # Convert 5-minute vehicle count to rate per second (capped at MAX_ARRIVAL_RATE)
            if count > 0:
                rate = round(min(MAX_ARRIVAL_RATE, max(0.01, count / 300.0)), 4)
            else:
                rate = 0.03  # default baseline flow
            dataset.append({"time": time_sec, "direction": direction, "rate": rate})

    return sorted(dataset, key=lambda row: (row["direction"], row["time"]))


def dataset_to_arrival_rates(dataset: list[dict], window_start: float, window_end: float) -> dict[str, float]:
    """Average matching direction rates for rows starting inside the requested time window."""
    totals = {direction: 0.0 for direction in DIRECTIONS}
    counts = {direction: 0 for direction in DIRECTIONS}
    for row in dataset:
        if window_start <= row["time"] < window_end and row["direction"] in totals:
            totals[row["direction"]] += float(row["rate"])
            counts[row["direction"]] += 1
    return {direction: totals[direction] / counts[direction] if counts[direction] else 0.0 for direction in DIRECTIONS}


def dataset_to_full_arrivals(dataset: list[dict], seed: int = 42,
                             duration: float = MAX_SIMULATION_SECONDS) -> dict[str, list[float]]:
    """Sample each piecewise-constant demand segment into sorted seeded vehicle timestamps."""
    if not math.isfinite(float(duration)) or duration < 0 or duration > MAX_SIMULATION_SECONDS:
        raise ValueError("duration must be between zero and 3600 seconds")
    rng = np.random.default_rng(seed)
    arrivals = {direction: [] for direction in DIRECTIONS}
    grouped = {direction: [] for direction in DIRECTIONS}
    for row in dataset:
        direction = row["direction"]
        start_time = float(row["time"])
        rate = float(row["rate"])
        if direction not in grouped or not math.isfinite(start_time) or not 0 <= start_time <= duration:
            raise ValueError("traffic rows must use a valid direction and in-range time")
        if not math.isfinite(rate) or not 0 <= rate <= MAX_ARRIVAL_RATE:
            raise ValueError(f"arrival rate must be between zero and {MAX_ARRIVAL_RATE} vehicles per second")
        grouped[direction].append((start_time, rate))

    for direction, segments in grouped.items():
        ordered_segments = sorted(segments)
        for index, (start_time, rate) in enumerate(ordered_segments):
            segment_end = ordered_segments[index + 1][0] if index + 1 < len(ordered_segments) else duration
            if rate <= 0 or segment_end <= start_time:
                continue
            window = segment_end - start_time
            expected_count = int(rng.poisson(rate * window))
            if expected_count > 0:
                timestamps = start_time + rng.uniform(0.0, window, size=expected_count)
                arrivals[direction].extend(float(ts) for ts in timestamps)
        arrivals[direction].sort()
    return arrivals
