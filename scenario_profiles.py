"""Shared piecewise demand profiles used by simulation, training, and evaluation."""

from __future__ import annotations

import math


SIMULATION_DURATION = 3600.0
MAX_ARRIVAL_RATE = 0.5

PRESET_PROFILES = {
    "balanced": {
        "N": [(0, 0.06)], "S": [(0, 0.06)],
        "E": [(0, 0.06)], "W": [(0, 0.06)],
    },
    "morning_peak": {
        "N": [(0, 0.145), (1800, 0.05)],
        "S": [(0, 0.14), (1800, 0.05)],
        "E": [(0, 0.04), (1800, 0.06)],
        "W": [(0, 0.04), (1800, 0.06)],
    },
    "evening_peak": {
        "N": [(0, 0.05)], "S": [(0, 0.05)],
        "E": [(0, 0.145), (1800, 0.05)],
        "W": [(0, 0.14), (1800, 0.05)],
    },
    "ns_skew": {
        "N": [(0, 0.145)], "S": [(0, 0.14)],
        "E": [(0, 0.035)], "W": [(0, 0.035)],
    },
    "ns_heavy": {
        "N": [(0, 0.145)], "S": [(0, 0.14)],
        "E": [(0, 0.035)], "W": [(0, 0.035)],
    },
    "heavy": {
        "N": [(0, 0.145)], "S": [(0, 0.145)],
        "E": [(0, 0.145)], "W": [(0, 0.145)],
    },
    "spike": {
        "N": [(0, 0.06)], "S": [(0, 0.06)],
        "E": [(0, 0.06)], "W": [(0, 0.06), (900, 0.25), (1200, 0.06)],
    },
    "t_junction": {
        "N": [(0, 0.08)], "S": [(0, 0.08)],
        "E": [(0, 0.10)],
    },
}


def profile_to_dataset(profile: dict) -> list[dict]:
    """Convert direction rate breakpoints into sorted time, direction, and rate rows."""
    rows = [
        {"time": float(start), "direction": direction, "rate": float(rate)}
        for direction, segments in profile.items()
        for start, rate in segments
    ]
    return sorted(rows, key=lambda row: (row["time"], row["direction"]))


def apply_stochastic_events(dataset: list[dict], events: list[dict], duration: float) -> list[dict]:
    """Split demand at event boundaries and apply active rate multipliers per direction."""
    directions = ("N", "S", "E", "W")
    rows_by_direction = {
        direction: sorted((row for row in dataset if row["direction"] == direction), key=lambda row: row["time"])
        for direction in directions
    }
    output = []

    for direction in directions:
        boundaries = {0.0, float(duration)}
        boundaries.update(float(row["time"]) for row in rows_by_direction[direction])
        for event in events:
            if event["direction"] == direction:
                start = max(0.0, float(event["time"]))
                end = min(float(duration), start + float(event["duration"]))
                boundaries.update((start, end))

        ordered = sorted(point for point in boundaries if 0 <= point < duration)
        for start in ordered:
            base_rate = 0.0
            for row in rows_by_direction[direction]:
                if row["time"] > start:
                    break
                base_rate = float(row["rate"])
            rate = base_rate
            for event in events:
                event_start = float(event["time"])
                if event["direction"] == direction and event_start <= start < event_start + float(event["duration"]):
                    rate *= float(event["multiplier"])
            output.append({
                "time": start,
                "direction": direction,
                "rate": min(MAX_ARRIVAL_RATE, rate),
            })

    return sorted(output, key=lambda row: (row["time"], row["direction"]))


def validate_profile_dataset(dataset: list[dict]) -> None:
    """Reject invalid rates and directions before generating a vehicle schedule."""
    for row in dataset:
        rate = float(row["rate"])
        start = float(row["time"])
        if row["direction"] not in ("N", "S", "E", "W"):
            raise ValueError("direction must be N, S, E, or W")
        if not math.isfinite(rate) or not 0 <= rate <= MAX_ARRIVAL_RATE:
            raise ValueError(f"arrival rate must be between 0 and {MAX_ARRIVAL_RATE} vehicles per second")
        if not math.isfinite(start) or not 0 <= start <= SIMULATION_DURATION:
            raise ValueError("time must be between zero and 3600 seconds")


def lanes_for_preset(scenario_name: str) -> dict[str, int]:
    """Return configured lanes per approach for a preset scenario."""
    if scenario_name in PRESET_PROFILES:
        return {d: 1 for d in PRESET_PROFILES[scenario_name]}
    return {"N": 1, "S": 1, "E": 1, "W": 1}


def sample_arrival_schedule(dataset: list[dict], seed: int | None, duration: float) -> dict[str, list[float]]:
    """Generate arrival timestamps from a rate dataset using a Poisson process."""
    import utilities
    rng = utilities.create_rng(seed)
    arrivals = {direction: [] for direction in ("N", "S", "E", "W")}
    rows_by_direction = {}
    for row in dataset:
        direction = row["direction"]
        if direction not in rows_by_direction:
            rows_by_direction[direction] = []
        rows_by_direction[direction].append(row)
    for direction in ("N", "S", "E", "W"):
        if direction not in rows_by_direction:
            continue
        direction_rows = sorted(rows_by_direction[direction], key=lambda r: r["time"])
        for i, row in enumerate(direction_rows):
            start_time = row["time"]
            rate = row["rate"]
            end_time = direction_rows[i + 1]["time"] if i + 1 < len(direction_rows) else duration
            segment_arrivals = utilities.generate_poisson_arrivals(rate, end_time - start_time, rng)
            arrivals[direction].extend(start_time + t for t in segment_arrivals)
    return {direction: sorted(arrivals[direction]) for direction in arrivals}
