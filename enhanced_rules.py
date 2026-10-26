"""Apply signal safety rules, record queue-aware decisions, and generate seeded demand events."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from simulation import SignalPhase


class SophisticatedRuleLayer:
    """Mask unsafe signal actions and record demand-driven and starvation-driven changes."""

    def __init__(self, min_green: int = 10, max_green: int = 60):
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.reset_episode()

    def get_legal_actions(self, traffic_light) -> list[int]:
        """Return keep or switch actions the signal can accept on its next simulation step."""
        if traffic_light.current_phase not in (SignalPhase.NORTH_SOUTH_GREEN, SignalPhase.EAST_WEST_GREEN):
            return [0]
        next_green_age = traffic_light.time_in_phase + 1
        if next_green_age >= self.max_green:
            return [1]
        if next_green_age < self.min_green:
            return [0]
        return [0, 1]

    def _phase_queues(self, queue_lengths: Dict[str, int], phase: int, traffic_light=None) -> tuple[int, int, str]:
        """Return active queue, waiting queue, and the waiting phase name for a green group."""
        if traffic_light is not None:
            active_dirs = traffic_light.get_green_directions()
        elif phase == SignalPhase.NORTH_SOUTH_GREEN:
            active_dirs = ("N", "S")
        else:
            active_dirs = ("E", "W")
        active = sum(queue_lengths.get(d, 0) for d in active_dirs)
        waiting = sum(queue_lengths.get(d, 0) for d in queue_lengths if d not in active_dirs)
        waiting_phase = "EW" if set(active_dirs) == {"N", "S"} else "NS"
        return active, waiting, waiting_phase

    def update_dynamic_thresholds(self, queue_lengths: Dict[str, int]) -> None:
        """Lower the queue gap and waiting limit as total demand grows."""
        total_queue = sum(queue_lengths.values())
        if total_queue >= 20:
            self.dynamic_thresholds = {"switch_threshold": 2, "starvation_limit": 30}
        elif total_queue >= 10:
            self.dynamic_thresholds = {"switch_threshold": 4, "starvation_limit": 40}
        else:
            self.dynamic_thresholds = {"switch_threshold": 6, "starvation_limit": 45}

    def validate_action(self, state: Tuple[int, int, int, int, int, int], action: int,
                        traffic_light, queue_lengths: Dict[str, int]) -> int:
        """Keep legal policy actions, force maximum-green changes, and serve a starved queued phase."""
        del state
        if action not in (0, 1):
            raise ValueError("action must be zero or one")
        if traffic_light.current_phase not in (SignalPhase.NORTH_SOUTH_GREEN, SignalPhase.EAST_WEST_GREEN):
            return 0

        self._update_moving_averages(queue_lengths)
        self.update_dynamic_thresholds(queue_lengths)
        active_queue, waiting_queue, waiting_phase = self._phase_queues(queue_lengths, traffic_light.current_phase, traffic_light)
        if waiting_queue:
            self.phase_starvation_counters[waiting_phase] += 1
        else:
            self.phase_starvation_counters[waiting_phase] = 0
        legal_actions = self.get_legal_actions(traffic_light)
        if action not in legal_actions:
            self.violations_prevented += 1
            action = 0

        next_green_age = traffic_light.time_in_phase + 1
        if next_green_age >= self.max_green:
            return 1
        waiting_time = self.phase_starvation_counters[waiting_phase]
        if action == 0 and waiting_queue > 0 and waiting_time >= self.dynamic_thresholds["starvation_limit"]:
            self.starvation_preventions += 1
            self.switches_allowed += 1
            return 1
        if action == 1:
            if waiting_queue == 0:
                self.violations_prevented += 1
                return 0
            if active_queue > waiting_queue + self.dynamic_thresholds["switch_threshold"] and waiting_time < self.dynamic_thresholds["starvation_limit"]:
                self.violations_prevented += 1
                return 0
            self.switches_allowed += 1
            if waiting_queue > active_queue:
                self.contextual_switches += 1
        return action

    def _update_moving_averages(self, queue_lengths: Dict[str, int]) -> None:
        """Keep the latest ten queue counts for each approach for run diagnostics."""
        for direction in ("N", "S", "E", "W"):
            history = self.moving_average_queues[direction]
            history.append(int(queue_lengths.get(direction, 0)))
            del history[:-10]

    def get_enhanced_statistics(self) -> Dict:
        """Return action-rule counts, current starvation counters, and queue moving averages."""
        statistics = {
            "violations_prevented": self.violations_prevented,
            "switches_allowed": self.switches_allowed,
            "contextual_switches": self.contextual_switches,
            "starvation_preventions": self.starvation_preventions,
            "violation_rate": self.violations_prevented / max(self.violations_prevented + self.switches_allowed, 1),
            "contextual_switch_rate": self.contextual_switches / max(self.switches_allowed, 1),
            "ns_starvation_time": self.phase_starvation_counters["NS"],
            "ew_starvation_time": self.phase_starvation_counters["EW"],
            "dynamic_thresholds": self.dynamic_thresholds.copy(),
        }
        for direction, values in self.moving_average_queues.items():
            if values:
                statistics[f"average_queue_{direction}"] = float(np.mean(values))
        return statistics

    def reset_episode(self) -> None:
        """Clear rule counts, queue histories, and time that each waiting phase has been starved."""
        self.moving_average_queues = {direction: [] for direction in ("N", "S", "E", "W")}
        self.phase_starvation_counters = {"NS": 0, "EW": 0}
        self.dynamic_thresholds = {"switch_threshold": 6, "starvation_limit": 45}
        self.violations_prevented = 0
        self.switches_allowed = 0
        self.contextual_switches = 0
        self.starvation_preventions = 0


class StochasticEventGenerator:
    """Generate reproducible temporary demand spikes and sensor-rate drops for training."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.events: list[dict] = []

    def generate_events(self, duration: int, base_rates: Dict[str, float]) -> List[Dict]:
        """Sample temporary directional rate changes inside an episode and return them by start time."""
        events = []
        for event_type, count, multiplier_range, duration_range in (
            ("spike", int(self.rng.poisson(3)), (2.0, 5.0), (30.0, 120.0)),
            ("failure", int(self.rng.poisson(1)), (0.1, 0.3), (60.0, 180.0)),
        ):
            for _ in range(count):
                direction = str(self.rng.choice(list(base_rates)))
                events.append({
                    "time": float(self.rng.uniform(0, duration)),
                    "type": event_type,
                    "direction": direction,
                    "multiplier": float(self.rng.uniform(*multiplier_range)),
                    "duration": float(self.rng.uniform(*duration_range)),
                })
        self.events = sorted(events, key=lambda event: event["time"])
        return [event.copy() for event in self.events]

    def get_modified_rates(self, base_rates: Dict[str, float], current_time: int) -> Dict[str, float]:
        """Return rates after multiplying all events active at the requested simulation time."""
        rates = base_rates.copy()
        for event in self.events:
            if event["time"] <= current_time < event["time"] + event["duration"]:
                rates[event["direction"]] *= event["multiplier"]
        return rates
