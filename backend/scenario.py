"""Build one-lane four-way scenarios from shared presets or validated rate rows."""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simulation
from backend.data_ingest import dataset_to_full_arrivals, generate_synthetic_dataset
from scenario_profiles import PRESET_PROFILES, SIMULATION_DURATION, profile_to_dataset


ROAD_CONFIGS = {
    name: {"lanes": {d: 1 for d in PRESET_PROFILES[name]}, "name": f"{name} intersection"}
    for name in PRESET_PROFILES
}
ROAD_CONFIGS["custom"] = {"lanes": {"N": 1, "S": 1, "E": 1, "W": 1}, "name": "Uploaded demand profile"}


class Scenario:
    """Keep a validated demand dataset so each requested seed can resample fresh arrivals."""

    def __init__(self, arrivals: dict, duration: int, seed: int, name: str,
                 dataset_summary: dict, road_config: dict | None = None,
                 dataset: list[dict] | None = None):
        self.arrivals = copy.deepcopy(arrivals)
        self.duration = int(duration)
        self.seed = int(seed)
        self.name = name
        self.dataset_summary = dict(dataset_summary)
        self.road_config = copy.deepcopy(road_config or ROAD_CONFIGS["custom"])
        self.dataset = copy.deepcopy(dataset or [])

    @classmethod
    def from_dataset(cls, dataset: list[dict], seed: int = 42, name: str = "custom") -> "Scenario":
        """Validate demand rows, sample one preview schedule, and retain the source profile for resampling."""
        normalized = [
            {"time": float(row["time"]), "direction": str(row["direction"]).upper(), "rate": float(row["rate"])}
            for row in dataset
        ]
        arrivals = dataset_to_full_arrivals(normalized, seed, SIMULATION_DURATION)
        estimate = sum(len(values) for values in arrivals.values())
        summary = {"total_rows": len(normalized), "estimated_vehicles": estimate}
        road_config = ROAD_CONFIGS.get(name, ROAD_CONFIGS["custom"])
        return cls(arrivals, int(SIMULATION_DURATION), seed, name, summary, road_config, normalized)

    @classmethod
    def from_preset(cls, name: str, seed: int = 42) -> "Scenario":
        """Create a named shared preset with a repeatable preview schedule."""
        if name not in PRESET_PROFILES:
            raise ValueError(f"Unknown preset: {name}")
        return cls.from_dataset(profile_to_dataset(PRESET_PROFILES[name]), seed, name)

    def to_intersection(self, arrival_seed: int | None = None) -> simulation.Intersection:
        """Create an independent simulator, optionally sampling new arrivals from the retained rates."""
        seed = self.seed if arrival_seed is None else int(arrival_seed)
        arrivals = self.arrivals
        if arrival_seed is not None:
            arrivals = dataset_to_full_arrivals(self.dataset, seed, self.duration)
        approaches = tuple(self.road_config.get("lanes", {}).keys()) or tuple(arrivals.keys())
        return simulation.Intersection(
            {direction: 0.0 for direction in approaches},
            seed=seed,
            lanes=self.road_config["lanes"],
            arrival_times=arrivals,
            duration=self.duration,
            approaches=approaches,
        )
