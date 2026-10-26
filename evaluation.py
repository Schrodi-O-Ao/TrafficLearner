"""Compare tabular control with fixed-time and queue-actuated baselines on paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np

import simulation
from backend.scenario import Scenario
from baseline_controller import FixedTimeController, QueueActuatedController
from enhanced_rules import SophisticatedRuleLayer
from scenario_profiles import PRESET_PROFILES
from tabular_agent import SimpleQLearningAgent


def _legal_actions(traffic_light) -> list[int]:
    """Return keep or switch actions accepted by the signal on the next one-second step."""
    if traffic_light.current_phase not in (0, 1):
        return [0]
    next_green_age = traffic_light.time_in_phase + 1
    if next_green_age >= traffic_light.maximum_green:
        return [1]
    return [0] if next_green_age < traffic_light.minimum_green else [0, 1]


def _run_one(intersection: simulation.Intersection, controller, policy: SimpleQLearningAgent | None = None) -> dict:
    """Advance one controller and report queue-area delay, served vehicles, backlog, and signal changes."""
    rules = SophisticatedRuleLayer() if policy is not None else None
    if rules:
        rules.reset_episode()
    while intersection.current_time < intersection.duration:
        queues = {direction: len(queue) for direction, queue in intersection.queues.items()}
        if policy is not None:
            state = intersection.get_state()
            action = policy.get_action(state, legal=_legal_actions(intersection.traffic_light), training=False)
            action = rules.validate_action(state, action, intersection.traffic_light, queues)
        elif isinstance(controller, QueueActuatedController):
            action = controller.get_action(intersection.traffic_light, queues)
        else:
            action = controller.get_action(intersection.traffic_light)
        intersection.step(action)

    statistics = intersection.get_statistics()
    duration = max(intersection.current_time, 1.0)
    return {
        "average_wait_experienced_seconds": statistics["average_wait_experienced"],
        "average_wait_completed_seconds": statistics["average_wait_completed"],
        "average_queue_length": intersection.queue_seconds / duration,
        "arrived": statistics["total_arrived"],
        "departed": statistics["total_departed"],
        "ending_queue": statistics["total_queue"],
        "signal_switches": statistics["signal_switches"],
        "throughput_per_minute": statistics["total_departed"] / (duration / 60.0),
        "steps": int(duration / intersection.time_step_seconds),
    }


def _summary(values: list[float]) -> dict:
    """Return arithmetic mean, sample spread, and observed range for paired run values."""
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def evaluate(scenarios: list[str], episode_length: int, runs: int, seed_start: int,
             policy_path: str, fixed_green: int = 15, queue_threshold: int = 3) -> dict:
    """Run each controller on identical schedules and summarize per-seed wait differences."""
    policy = SimpleQLearningAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
    policy.load(policy_path)
    policy.epsilon = 0.0
    all_scenarios = {}
    for scenario_name in scenarios:
        scenario = Scenario.from_preset(scenario_name, seed_start)
        if not 1 <= episode_length <= scenario.duration:
            raise ValueError(f"episode_length must be between one second and {scenario.duration}")
        paired_runs = []
        for run_index in range(runs):
            seed = seed_start + run_index
            controller_results = {}
            for controller_name in ("fixed_time", "queue_actuated", "q_learning"):
                intersection = scenario.to_intersection(arrival_seed=seed)
                intersection.duration = float(episode_length)
                if controller_name == "fixed_time":
                    controller = FixedTimeController(green_time=fixed_green)
                    controller_results[controller_name] = _run_one(intersection, controller)
                elif controller_name == "queue_actuated":
                    controller = QueueActuatedController(threshold=queue_threshold)
                    controller_results[controller_name] = _run_one(intersection, controller)
                else:
                    controller_results[controller_name] = _run_one(intersection, None, policy)
            paired_runs.append({"seed": seed, **controller_results})

        wait_differences_fixed = [
            row["q_learning"]["average_wait_experienced_seconds"] - row["fixed_time"]["average_wait_experienced_seconds"]
            for row in paired_runs
        ]
        wait_differences_actuated = [
            row["q_learning"]["average_wait_experienced_seconds"] - row["queue_actuated"]["average_wait_experienced_seconds"]
            for row in paired_runs
        ]
        all_scenarios[scenario_name] = {
            "episode_length_seconds": episode_length,
            "run_count": runs,
            "seed_range": [seed_start, seed_start + runs - 1],
            "controllers": {
                controller_name: {
                    metric_name: _summary([row[controller_name][metric_name] for row in paired_runs])
                    for metric_name in paired_runs[0][controller_name]
                }
                for controller_name in ("fixed_time", "queue_actuated", "q_learning")
            },
            "paired_wait_difference_q_learning_minus_fixed_time_seconds": _summary(wait_differences_fixed),
            "paired_wait_difference_q_learning_minus_queue_actuated_seconds": _summary(wait_differences_actuated),
            "paired_runs": paired_runs,
        }
    return {
        "policy_path": os.path.abspath(policy_path),
        "fixed_green_seconds": fixed_green,
        "queue_actuated_threshold_vehicles": queue_threshold,
        "scenarios": all_scenarios,
    }


def save_results(results: dict, output_dir: str) -> None:
    """Write paired seed measurements and the experiment settings as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(os.path.join(output_dir, "paired_results.csv"), "w", newline="", encoding="utf-8") as file:
        columns = ["scenario", "seed", "controller", "average_wait_experienced_seconds",
                   "average_wait_completed_seconds", "average_queue_length", "arrived", "departed",
                   "ending_queue", "signal_switches", "throughput_per_minute", "steps"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for scenario_name, scenario_result in results["scenarios"].items():
            for run in scenario_result["paired_runs"]:
                for controller_name in ("fixed_time", "queue_actuated", "q_learning"):
                    writer.writerow({"scenario": scenario_name, "seed": run["seed"],
                                     "controller": controller_name, **run[controller_name]})


def main() -> None:
    """Run paired evaluation seeds and print the measured Q-learning wait differences."""
    parser = argparse.ArgumentParser(description="Compare learned and rule-based traffic signal controllers")
    parser.add_argument("--agent", default=os.path.join("models", "tabular_policy.json"), help="Portable JSON Q-table")
    parser.add_argument("--scenarios", nargs="+", default=["heavy"], choices=sorted(PRESET_PROFILES))
    parser.add_argument("--episode_length", type=int, default=3600, help="Measured episode duration in seconds")
    parser.add_argument("--runs", type=int, default=30, help="Number of paired seeds")
    parser.add_argument("--seed", type=int, default=42, help="First held-out evaluation seed")
    parser.add_argument("--fixed_green", type=int, default=15, help="Tuned fixed-time green interval in seconds")
    parser.add_argument("--queue_threshold", type=int, default=3, help="Queue gap required by actuated control")
    parser.add_argument("--output_dir", default=None, help="Directory for paired run data")
    args = parser.parse_args()
    if not 1 <= args.episode_length <= 3600 or args.fixed_green < 1 or args.queue_threshold < 0 or args.runs < 1:
        parser.error("episode length, runs, fixed green, and queue threshold are outside their valid ranges")
    output_dir = args.output_dir or os.path.join("output", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results = evaluate(args.scenarios, args.episode_length, args.runs, args.seed,
                       args.agent, args.fixed_green, args.queue_threshold)
    save_results(results, output_dir)
    for scenario_name, scenario_result in results["scenarios"].items():
        fixed_difference = scenario_result["paired_wait_difference_q_learning_minus_fixed_time_seconds"]
        actuated_difference = scenario_result["paired_wait_difference_q_learning_minus_queue_actuated_seconds"]
        print(f"{scenario_name} ({scenario_result['run_count']} paired seeds, {args.episode_length} s each)")
        print(f"  Q-learning minus fixed-time experienced wait: {fixed_difference['mean']:.3f} s "
              f"(SD {fixed_difference['sample_standard_deviation']:.3f} s)")
        print(f"  Q-learning minus queue-actuated experienced wait: {actuated_difference['mean']:.3f} s "
              f"(SD {actuated_difference['sample_standard_deviation']:.3f} s)")
    print(f"Paired measurements saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
