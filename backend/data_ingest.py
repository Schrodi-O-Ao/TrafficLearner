"""Validate uploaded traffic rate rows and sample reproducible vehicle arrival schedules."""

from __future__ import annotations

import csv
import io
import math

import numpy as np


MAX_DATASET_ROWS = 1000
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


def load_from_csv(filepath_or_bytes) -> list[dict]:
    """Read a UTF-8 CSV with time, direction, and rate or 300-second vehicle count columns."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        source = io.StringIO(filepath_or_bytes.decode("utf-8-sig"))
        close_source = False
    else:
        source = open(filepath_or_bytes, "r", encoding="utf-8-sig", newline="")
        close_source = True

    try:
        reader = csv.DictReader(source)
        headers = {header.strip().lower() for header in (reader.fieldnames or [])}
        if not ({"time_seconds", "time"} & headers) or "direction" not in headers:
            raise ValueError("CSV headers must include time_seconds or time, plus direction.")
        if not ({"arrival_rate", "vehicle_count"} & headers):
            raise ValueError("CSV headers must include arrival_rate or vehicle_count.")
        dataset = []
        for row_number, original_row in enumerate(reader, start=2):
            if row_number > MAX_DATASET_ROWS + 1:
                raise ValueError(f"CSV may contain at most {MAX_DATASET_ROWS} data rows.")
            row = {(key or "").strip().lower(): value for key, value in original_row.items()}
            raw_time = row.get("time_seconds") or row.get("time")
            try:
                time_seconds = float(raw_time)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number}: time_seconds must be a number.") from error
            if not math.isfinite(time_seconds) or not 0 <= time_seconds <= MAX_SIMULATION_SECONDS:
                raise ValueError(f"Row {row_number}: time_seconds must be between 0 and 3600.")

            direction = (row.get("direction") or "").strip().upper()
            if direction not in DIRECTIONS:
                raise ValueError(f"Row {row_number}: direction must be N, S, E, or W.")
            try:
                if row.get("arrival_rate") not in (None, ""):
                    rate = float(row["arrival_rate"])
                elif row.get("vehicle_count") not in (None, ""):
                    rate = float(row["vehicle_count"]) / 300.0
                else:
                    raise ValueError("missing arrival_rate or vehicle_count")
            except ValueError as error:
                raise ValueError(f"Row {row_number}: arrival rate or vehicle count must be numeric.") from error
            if not math.isfinite(rate) or not 0 <= rate <= MAX_ARRIVAL_RATE:
                raise ValueError(f"Row {row_number}: arrival rate must be between 0 and {MAX_ARRIVAL_RATE} vehicles per second.")
            dataset.append({"time": time_seconds, "direction": direction, "rate": rate})
    finally:
        if close_source:
            source.close()

    if not dataset:
        raise ValueError("CSV has no traffic rows to simulate.")
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
            time_seconds = start_time
            while True:
                time_seconds += float(rng.exponential(1.0 / rate))
                if time_seconds >= segment_end:
                    break
                arrivals[direction].append(time_seconds)
        arrivals[direction].sort()
    return arrivals
