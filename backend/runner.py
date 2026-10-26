"""Advance paired controllers one shared simulation second at a time and emit real vehicle events."""

from __future__ import annotations

import os
from functools import lru_cache

from baseline_controller import FixedTimeController
from enhanced_rules import SophisticatedRuleLayer
from tabular_agent import SimpleQLearningAgent


DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tabular_policy.json")


@lru_cache(maxsize=4)
def load_policy(policy_path: str = DEFAULT_POLICY_PATH) -> SimpleQLearningAgent:
    """Load a portable learned Q-table once so web requests do not train agents."""
    agent = SimpleQLearningAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
    agent.load(policy_path)
    agent.epsilon = 0.0
    return agent


def _queue_lengths(intersection) -> dict[str, int]:
    """Return vehicle counts for each approach without exposing mutable simulator queues."""
    return {direction: len(queue) for direction, queue in intersection.queues.items()}


def _legal_actions(traffic_light) -> list[int]:
    """Mask signal actions that cannot be accepted during clearance or minimum green."""
    if traffic_light.current_phase not in (0, 1):
        return [0]
    next_green_age = traffic_light.time_in_phase + 1
    if next_green_age >= traffic_light.maximum_green:
        return [1]
    if next_green_age < traffic_light.minimum_green:
        return [0]
    return [0, 1]


def _controller_snapshot(intersection, action: int, departed: int) -> dict:
    """Package one controller's post-step queues, phase, service count, and generated vehicle events."""
    return {
        "queues": _queue_lengths(intersection),
        "queue_vehicles": {
            direction: [{"vehicle_id": vehicle.vehicle_id, "arrival_time": vehicle.arrived_at,
                         "lane_index": vehicle.lane_index}
                        for vehicle in queue]
            for direction, queue in intersection.queues.items()
        },
        "phase": intersection.traffic_light.current_phase,
        "departed": departed,
        "action": action,
        "events": [dict(event) for event in intersection.events_this_step],
    }


def _metrics(intersection) -> dict:
    """Calculate queue-area delay, completed delay, throughput, backlog, and green changes."""
    statistics = intersection.get_statistics()
    minutes = max(intersection.current_time / 60.0, 1.0 / 60.0)
    return {
        "average_wait_experienced": round(statistics["average_wait_experienced"], 3),
        "average_wait_completed": round(statistics["average_wait_completed"], 3),
        "average_queue_length": round(intersection.queue_seconds / max(intersection.current_time, 1.0), 3),
        "total_arrived": statistics["total_arrived"],
        "total_departed": statistics["total_departed"],
        "ending_queue": statistics["total_queue"],
        "throughput_per_minute": round(statistics["total_departed"] / minutes, 3),
        "total_switches": statistics["signal_switches"],
    }


def stream_comparison(scenario, episode_length: int = 3600, arrival_seed: int | None = None,
                      policy_path: str = DEFAULT_POLICY_PATH):
    """Yield matched fixed-time and learned states after each shared simulation second."""
    if not 1 <= episode_length <= scenario.duration:
        raise ValueError("episode_length must be between one second and the scenario duration")
    seed = scenario.seed if arrival_seed is None else int(arrival_seed)
    baseline_intersection = scenario.to_intersection(arrival_seed=seed)
    learning_intersection = scenario.to_intersection(arrival_seed=seed)
    policy = load_policy(policy_path)
    fixed_controller = FixedTimeController(green_time=15)
    rule_layer = SophisticatedRuleLayer()
    rule_layer.reset_episode()
    expected_arrivals = sum(
        sum(timestamp < episode_length for timestamp in schedule)
        for schedule in baseline_intersection.arrival_times.values()
    )
    yield {
        "type": "sim_start",
        "t": 0.0,
        "duration": episode_length,
        "total_vehicles_expected": expected_arrivals,
    }

    while baseline_intersection.current_time < episode_length:
        baseline_queues = _queue_lengths(baseline_intersection)
        learning_queues = _queue_lengths(learning_intersection)
        learning_state = learning_intersection.get_state()
        legal_actions = _legal_actions(learning_intersection.traffic_light)
        learning_action = policy.get_action(learning_state, legal=legal_actions, training=False)
        learning_action = rule_layer.validate_action(
            learning_state, learning_action, learning_intersection.traffic_light, learning_queues
        )
        baseline_action = fixed_controller.get_action(baseline_intersection.traffic_light)
        _, baseline_departed, _ = baseline_intersection.step(baseline_action)
        _, learning_departed, _ = learning_intersection.step(learning_action)
        timestamp = baseline_intersection.current_time
        yield {
            "type": "tick",
            "t": timestamp,
            "controllers": {
                "baseline": _controller_snapshot(baseline_intersection, baseline_action, baseline_departed),
                "rl": _controller_snapshot(learning_intersection, learning_action, learning_departed),
            },
        }

    baseline_metrics = _metrics(baseline_intersection)
    learning_metrics = _metrics(learning_intersection)
    yield {
        "type": "comparison_ready",
        "t": float(episode_length),
        "baseline": baseline_metrics,
        "rl": learning_metrics,
        "rule_layer": rule_layer.get_enhanced_statistics(),
    }


def run_comparison(scenario, episode_length: int = 3600, arrival_seed: int | None = None,
                   policy_path: str = DEFAULT_POLICY_PATH) -> dict:
    """Collect a lockstep replay and per-controller event logs for reports and metrics routes."""
    messages = list(stream_comparison(scenario, episode_length, arrival_seed, policy_path))
    ready = messages[-1]
    start = messages[0]
    baseline_events = [{"type": "sim_start", "t": start["t"], "duration": start["duration"]}]
    learning_events = [dict(baseline_events[0], controller="rl")]
    for message in messages[1:-1]:
        timestamp = message["t"]
        for controller, events in (("baseline", baseline_events), ("rl", learning_events)):
            snapshot = message["controllers"][controller]
            events.extend(dict(event, controller=controller) for event in snapshot["events"])
            events.append({
                "type": "step", "t": timestamp, "controller": controller,
                "queues": snapshot["queues"], "phase": snapshot["phase"],
                "departed": snapshot["departed"], "action": snapshot["action"],
            })
    baseline_events.append({"type": "sim_done", "t": ready["t"], "metrics": ready["baseline"]})
    learning_events.append({"type": "sim_done", "t": ready["t"], "metrics": ready["rl"]})
    return {
        "baseline": baseline_events,
        "rl": learning_events,
        "interleaved": messages,
        "comparison": ready,
    }
