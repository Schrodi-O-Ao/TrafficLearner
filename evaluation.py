"""Compare tabular control with fixed-time and queue-actuated baselines on paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _run_one(intersection: simulation.Intersection, controller,
             policy: SimpleQLearningAgent | None = None,
             use_rules: bool = True) -> dict:
    """Advance one controller and report queue-area delay, served vehicles, backlog, and signal changes.

    When *policy* is provided and *use_rules* is True the rule layer wraps
    the learned actions (the default hybrid mode).  Setting *use_rules* to
    False runs the raw Q-table without the rule layer for ablation.
    """
    rules = SophisticatedRuleLayer() if (policy is not None and use_rules) else None
    if rules:
        rules.reset_episode()
    while intersection.current_time < intersection.duration:
        queues = {direction: len(queue) for direction, queue in intersection.queues.items()}
        if policy is not None:
            state = intersection.get_state()
            action = policy.get_action(state, legal=_legal_actions(intersection.traffic_light), training=False)
            if rules is not None:
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


def _run_rules_only(intersection: simulation.Intersection) -> dict:
    """Run the rule layer alone by feeding it a constant switch-request and letting it decide."""
    rules = SophisticatedRuleLayer()
    rules.reset_episode()
    while intersection.current_time < intersection.duration:
        queues = {direction: len(queue) for direction, queue in intersection.queues.items()}
        state = intersection.get_state()
        # Always request a switch; the rule layer decides whether it is safe.
        action = rules.validate_action(state, 1, intersection.traffic_light, queues)
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


def _paired_statistics(differences: list[float]) -> dict:
    """Return mean difference, SD, 95% confidence interval, and paired t-test p-value."""
    n = len(differences)
    mean_diff = float(np.mean(differences))
    if n < 2:
        return {"mean": mean_diff, "sample_standard_deviation": 0.0,
                "ci_95_lower": mean_diff, "ci_95_upper": mean_diff, "t_statistic": 0.0, "p_value": 1.0}
    sd = float(np.std(differences, ddof=1))
    se = sd / math.sqrt(n)
    # Two-tailed t critical value for 95% CI (approximation for df >= 20 uses 2.093 for df=19,
    # but we compute it properly using the percent-point function from scipy if available,
    # otherwise fall back to a conservative 2.093 for df <= 20 and 1.96 for large n).
    df = n - 1
    if df >= 30:
        t_crit = 1.96
    elif df >= 20:
        t_crit = 2.086
    else:
        # Small-sample critical values for common df (two-tailed, alpha=0.05)
        _t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}
        t_crit = _t_table.get(df, 2.093)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    t_stat = mean_diff / se if se > 0 else 0.0
    # Two-tailed p-value approximation using the normal distribution for |t| > 1
    if abs(t_stat) < 1e-12:
        p_value = 1.0
    else:
        try:
            from scipy.stats import t as t_dist
            p_value = float(2.0 * t_dist.sf(abs(t_stat), df))
        except ImportError:
            # Rough two-tailed approximation without scipy
            z = abs(t_stat)
            p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
    return {
        "mean": mean_diff,
        "sample_standard_deviation": sd,
        "standard_error": float(se),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "t_statistic": float(t_stat),
        "p_value": p_value,
        "n": n,
    }


# The five controller names used in full (ablation) evaluation mode.
ABLATION_CONTROLLERS = ("fixed_time", "queue_actuated", "q_learning_raw", "rules_only", "q_learning")
# The three controller names used in standard evaluation mode.
STANDARD_CONTROLLERS = ("fixed_time", "queue_actuated", "q_learning")


def evaluate(scenarios: list[str], episode_length: int, runs: int, seed_start: int,
             policy_path: str, fixed_green: int = 15, queue_threshold: int = 3,
             ablation: bool = False) -> dict:
    """Run each controller on identical schedules and summarize per-seed wait differences.

    When *ablation* is True two extra controllers are included:
    ``q_learning_raw`` runs the Q-table without the rule layer, and
    ``rules_only`` runs the rule layer with a constant switch request.
    """
    policy = SimpleQLearningAgent(epsilon_start=0.0, epsilon_min=0.0, seed=0)
    policy.load(policy_path)
    policy.epsilon = 0.0
    controller_names = ABLATION_CONTROLLERS if ablation else STANDARD_CONTROLLERS
    all_scenarios = {}
    for scenario_name in scenarios:
        scenario = Scenario.from_preset(scenario_name, seed_start)
        if not 1 <= episode_length <= scenario.duration:
            raise ValueError(f"episode_length must be between one second and {scenario.duration}")
        paired_runs = []
        for run_index in range(runs):
            seed = seed_start + run_index
            controller_results = {}
            for controller_name in controller_names:
                intersection = scenario.to_intersection(arrival_seed=seed)
                intersection.duration = float(episode_length)
                if controller_name == "fixed_time":
                    controller = FixedTimeController(green_time=fixed_green)
                    controller_results[controller_name] = _run_one(intersection, controller)
                elif controller_name == "queue_actuated":
                    controller = QueueActuatedController(threshold=queue_threshold)
                    controller_results[controller_name] = _run_one(intersection, controller)
                elif controller_name == "q_learning_raw":
                    controller_results[controller_name] = _run_one(intersection, None, policy, use_rules=False)
                elif controller_name == "rules_only":
                    controller_results[controller_name] = _run_rules_only(intersection)
                else:
                    controller_results[controller_name] = _run_one(intersection, None, policy, use_rules=True)
            paired_runs.append({"seed": seed, **controller_results})

        # Paired difference statistics for the hybrid Q-learning controller.
        hybrid_key = "q_learning"
        wait_differences_fixed = [
            row[hybrid_key]["average_wait_experienced_seconds"] - row["fixed_time"]["average_wait_experienced_seconds"]
            for row in paired_runs
        ]
        wait_differences_actuated = [
            row[hybrid_key]["average_wait_experienced_seconds"] - row["queue_actuated"]["average_wait_experienced_seconds"]
            for row in paired_runs
        ]
        scenario_result = {
            "episode_length_seconds": episode_length,
            "run_count": runs,
            "seed_range": [seed_start, seed_start + runs - 1],
            "controllers": {
                cn: {
                    metric_name: _summary([row[cn][metric_name] for row in paired_runs])
                    for metric_name in paired_runs[0][cn]
                }
                for cn in controller_names
            },
            "paired_wait_difference_q_learning_minus_fixed_time_seconds": _paired_statistics(wait_differences_fixed),
            "paired_wait_difference_q_learning_minus_queue_actuated_seconds": _paired_statistics(wait_differences_actuated),
            "paired_runs": paired_runs,
        }

        # If ablation mode, also add differences for the raw policy and rules-only controllers.
        if ablation:
            raw_diffs_fixed = [
                row["q_learning_raw"]["average_wait_experienced_seconds"] - row["fixed_time"]["average_wait_experienced_seconds"]
                for row in paired_runs
            ]
            raw_diffs_actuated = [
                row["q_learning_raw"]["average_wait_experienced_seconds"] - row["queue_actuated"]["average_wait_experienced_seconds"]
                for row in paired_runs
            ]
            rules_diffs_fixed = [
                row["rules_only"]["average_wait_experienced_seconds"] - row["fixed_time"]["average_wait_experienced_seconds"]
                for row in paired_runs
            ]
            scenario_result["paired_wait_difference_q_learning_raw_minus_fixed_time_seconds"] = _paired_statistics(raw_diffs_fixed)
            scenario_result["paired_wait_difference_q_learning_raw_minus_queue_actuated_seconds"] = _paired_statistics(raw_diffs_actuated)
            scenario_result["paired_wait_difference_rules_only_minus_fixed_time_seconds"] = _paired_statistics(rules_diffs_fixed)

        all_scenarios[scenario_name] = scenario_result
    return {
        "policy_path": os.path.abspath(policy_path),
        "fixed_green_seconds": fixed_green,
        "queue_actuated_threshold_vehicles": queue_threshold,
        "ablation": ablation,
        "scenarios": all_scenarios,
    }


def save_results(results: dict, output_dir: str) -> None:
    """Write paired seed measurements and the experiment settings as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    controller_names = ABLATION_CONTROLLERS if results.get("ablation") else STANDARD_CONTROLLERS
    with open(os.path.join(output_dir, "paired_results.csv"), "w", newline="", encoding="utf-8") as file:
        columns = ["scenario", "seed", "controller", "average_wait_experienced_seconds",
                   "average_wait_completed_seconds", "average_queue_length", "arrived", "departed",
                   "ending_queue", "signal_switches", "throughput_per_minute", "steps"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for scenario_name, scenario_result in results["scenarios"].items():
            for run in scenario_result["paired_runs"]:
                for controller_name in controller_names:
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
    parser.add_argument("--ablation", action="store_true",
                        help="Include q_learning_raw (no rule layer) and rules_only controllers")
    args = parser.parse_args()
    if not 1 <= args.episode_length <= 3600 or args.fixed_green < 1 or args.queue_threshold < 0 or args.runs < 1:
        parser.error("episode length, runs, fixed green, and queue threshold are outside their valid ranges")
    output_dir = args.output_dir or os.path.join("output", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results = evaluate(args.scenarios, args.episode_length, args.runs, args.seed,
                       args.agent, args.fixed_green, args.queue_threshold, args.ablation)
    save_results(results, output_dir)
    for scenario_name, scenario_result in results["scenarios"].items():
        fixed_diff = scenario_result["paired_wait_difference_q_learning_minus_fixed_time_seconds"]
        actuated_diff = scenario_result["paired_wait_difference_q_learning_minus_queue_actuated_seconds"]
        print(f"{scenario_name} ({scenario_result['run_count']} paired seeds, {args.episode_length} s each)")
        print(f"  Q-learning minus fixed-time: {fixed_diff['mean']:+.2f} s "
              f"(SD {fixed_diff['sample_standard_deviation']:.2f}, "
              f"95% CI [{fixed_diff['ci_95_lower']:+.2f}, {fixed_diff['ci_95_upper']:+.2f}], "
              f"p={fixed_diff['p_value']:.4f})")
        print(f"  Q-learning minus queue-actuated: {actuated_diff['mean']:+.2f} s "
              f"(SD {actuated_diff['sample_standard_deviation']:.2f}, "
              f"95% CI [{actuated_diff['ci_95_lower']:+.2f}, {actuated_diff['ci_95_upper']:+.2f}], "
              f"p={actuated_diff['p_value']:.4f})")
        if args.ablation:
            raw_fixed = scenario_result["paired_wait_difference_q_learning_raw_minus_fixed_time_seconds"]
            raw_act = scenario_result["paired_wait_difference_q_learning_raw_minus_queue_actuated_seconds"]
            rules_fixed = scenario_result["paired_wait_difference_rules_only_minus_fixed_time_seconds"]
            print(f"  Q-raw minus fixed-time: {raw_fixed['mean']:+.2f} s "
                  f"(SD {raw_fixed['sample_standard_deviation']:.2f}, p={raw_fixed['p_value']:.4f})")
            print(f"  Q-raw minus queue-actuated: {raw_act['mean']:+.2f} s "
                  f"(SD {raw_act['sample_standard_deviation']:.2f}, p={raw_act['p_value']:.4f})")
            print(f"  Rules-only minus fixed-time: {rules_fixed['mean']:+.2f} s "
                  f"(SD {rules_fixed['sample_standard_deviation']:.2f}, p={rules_fixed['p_value']:.4f})")
    print(f"Paired measurements saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
