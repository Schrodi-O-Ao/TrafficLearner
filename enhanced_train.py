"""Train the bundled tabular signal policy on fresh shared demand profiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np

import simulation
from backend.data_ingest import dataset_to_full_arrivals
from enhanced_rules import SophisticatedRuleLayer, StochasticEventGenerator
from scenario_profiles import PRESET_PROFILES, apply_stochastic_events, profile_to_dataset
from tabular_agent import SimpleQLearningAgent


def _base_rates(scenario_name: str) -> dict[str, float]:
    """Read one approach's initial demand rate from the shared named profile."""
    if scenario_name not in PRESET_PROFILES:
        raise ValueError(f"Unknown training scenario: {scenario_name}")
    return {direction: segments[0][1] for direction, segments in PRESET_PROFILES[scenario_name].items()}


def _make_intersection(scenario_name: str, duration: int, seed: int, events: list[dict] | None = None) -> simulation.Intersection:
    """Resample a scenario's piecewise demand with optional temporary event multipliers."""
    source_dataset = profile_to_dataset(PRESET_PROFILES[scenario_name])
    if events:
        source_dataset = apply_stochastic_events(source_dataset, events, duration)
    arrivals = dataset_to_full_arrivals(source_dataset, seed, duration)
    return simulation.Intersection(_base_rates(scenario_name), seed=seed, arrival_times=arrivals, duration=duration)


def _train_episode(agent: SimpleQLearningAgent, scenario_name: str, duration: int,
                   seed: int, use_events: bool, rule_layer: SophisticatedRuleLayer | None = None) -> dict:
    """Train through one fresh episode and return simulator measures from the completed run."""
    base_rates = _base_rates(scenario_name)
    event_generator = StochasticEventGenerator(seed + 1_000_000)
    events = event_generator.generate_events(duration, base_rates) if use_events else []
    intersection = _make_intersection(scenario_name, duration, seed, events)
    active_rules = rule_layer or SophisticatedRuleLayer()
    active_rules.reset_episode()

    while intersection.current_time < duration:
        state = intersection.get_state()
        queues_before = {direction: len(queue) for direction, queue in intersection.queues.items()}
        legal_actions = active_rules.get_legal_actions(intersection.traffic_light)
        requested_action = agent.get_action(state, legal=legal_actions, training=True)
        action = active_rules.validate_action(state, requested_action, intersection.traffic_light, queues_before)
        intersection.step(action)
        next_state = intersection.get_state()
        next_legal_actions = active_rules.get_legal_actions(intersection.traffic_light)
        switch_cost = 4.0 if action == 1 else 0.0
        reward = (-sum(len(queue) for queue in intersection.queues.values()) - switch_cost) * intersection.time_step_seconds
        agent.update(state, action, reward, next_state,
                     done=intersection.current_time >= duration, next_legal=next_legal_actions)

    agent.decay_exploration()
    statistics = intersection.get_statistics()
    return {
        "scenario": scenario_name,
        "seed": seed,
        "duration_seconds": duration,
        "average_wait_experienced_seconds": statistics["average_wait_experienced"],
        "average_wait_completed_seconds": statistics["average_wait_completed"],
        "average_queue_length": intersection.queue_seconds / max(duration, 1),
        "arrived": statistics["total_arrived"],
        "departed": statistics["total_departed"],
        "ending_queue": statistics["total_queue"],
        "signal_switches": statistics["signal_switches"],
        "stochastic_events": len(events),
        "episode_reward_queue_seconds": -intersection.queue_seconds,
    }


def _evaluate_candidate(parameters: dict, scenario_names: list[str], duration: int, seed: int) -> float:
    """Train one hyperparameter candidate briefly and score it on separate fixed seeds."""
    agent = SimpleQLearningAgent(
        learning_rate=parameters["learning_rate"],
        epsilon_start=0.5,
        epsilon_decay=parameters["epsilon_decay"],
        gamma=parameters["gamma"],
        seed=seed,
    )
    training_duration = min(duration, 600)
    for episode in range(2):
        _train_episode(agent, scenario_names[episode % len(scenario_names)], training_duration,
                       seed + episode, use_events=False)
    holdout_costs = []
    for index, scenario_name in enumerate(scenario_names):
        intersection = _make_intersection(scenario_name, training_duration, seed + 1000 + index)
        rules = SophisticatedRuleLayer()
        while intersection.current_time < training_duration:
            state = intersection.get_state()
            legal = rules.get_legal_actions(intersection.traffic_light)
            action = agent.get_action(state, legal=legal, training=False)
            action = rules.validate_action(state, action, intersection.traffic_light,
                                          {direction: len(queue) for direction, queue in intersection.queues.items()})
            intersection.step(action)
        holdout_costs.append(intersection.queue_seconds / max(intersection.total_arrived, 1))
    return float(np.mean(holdout_costs))


def evolutionary_search(scenario_names: list[str], duration: int, seed: int,
                        population_size: int = 6, generations: int = 2) -> dict:
    """Select learning rate, exploration decay, and discount using seeded selection and mutation."""
    rng = np.random.default_rng(seed)
    population = [
        {"learning_rate": float(rng.uniform(0.05, 0.3)),
         "epsilon_decay": float(rng.uniform(0.97, 0.999)),
         "gamma": float(rng.uniform(0.90, 0.999))}
        for _ in range(population_size)
    ]
    best = None
    best_score = float("inf")
    for generation in range(generations):
        scored = []
        for index, candidate in enumerate(population):
            score = _evaluate_candidate(candidate, scenario_names, duration, seed + generation * 1000 + index * 10)
            scored.append((score, candidate))
            print(f"Search generation {generation + 1}, candidate {index + 1}: experienced wait {score:.2f} s")
        scored.sort(key=lambda row: row[0])
        if scored[0][0] < best_score:
            best_score, best = scored[0][0], scored[0][1].copy()
        parents = [candidate for _, candidate in scored[:2]]
        population = [candidate.copy() for candidate in parents]
        while len(population) < population_size:
            parent = parents[int(rng.integers(len(parents)))]
            child = {
                "learning_rate": float(np.clip(parent["learning_rate"] * rng.uniform(0.75, 1.25), 0.02, 0.5)),
                "epsilon_decay": float(np.clip(parent["epsilon_decay"] + rng.normal(0, 0.006), 0.95, 0.9999)),
                "gamma": float(np.clip(parent["gamma"] + rng.normal(0, 0.02), 0.85, 0.9999)),
            }
            population.append(child)
    print(f"Selected settings scored {best_score:.2f} s held-out experienced wait: {best}")
    return best


class EnhancedTrainingManager:
    """Run tabular learning, event training, CSV logging, and portable policy export."""

    def __init__(self, config: dict):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"tabular_train_{timestamp}"
        self.output_dir = os.path.join(config.get("output_root", "output"), self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.agent = SimpleQLearningAgent(
            learning_rate=config["learning_rate"],
            epsilon_start=config["epsilon_start"],
            epsilon_decay=config["epsilon_decay"],
            gamma=config["gamma"],
            seed=config["seed"],
        )
        self.rule_layer = SophisticatedRuleLayer()
        self.episode_results = []

    def train(self) -> SimpleQLearningAgent:
        """Train on disjoint numbered seeds and write metrics, configuration, and final JSON policy."""
        start_time = time.perf_counter()
        episode_count = self.config["episodes"]
        duration = self.config["episode_length"]
        scenarios = self.config["scenarios"]
        for episode in range(episode_count):
            scenario_name = scenarios[episode % len(scenarios)]
            seed = self.config["seed"] + episode
            result = _train_episode(self.agent, scenario_name, duration, seed, True, self.rule_layer)
            result["episode"] = episode + 1
            result["epsilon"] = self.agent.epsilon
            self.episode_results.append(result)
            if (episode + 1) % self.config["print_interval"] == 0 or episode == 0:
                recent = self.episode_results[-self.config["print_interval"]:]
                mean_wait = float(np.mean([row["average_wait_experienced_seconds"] for row in recent]))
                print(f"Episode {episode + 1}/{episode_count} | {scenario_name} | "
                      f"experienced wait {mean_wait:.2f} s | epsilon {self.agent.epsilon:.3f} | "
                      f"Q states {len(self.agent.q_table)}")

        policy_path = self.config["policy_output"]
        self.agent.save(policy_path)
        report = {
            "episodes": self.episode_results,
            "policy_path": os.path.abspath(policy_path),
            "agent": self.agent.get_statistics(),
            "settings": self.config,
            "elapsed_seconds": round(time.perf_counter() - start_time, 2),
            "training_seed_range": [self.config["seed"], self.config["seed"] + episode_count - 1],
            "evaluation_seed_range_reserved": [42, 71],
        }
        report_path = os.path.join(self.output_dir, "training_summary.json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        csv_path = os.path.join(self.output_dir, "episode_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.episode_results[0]) if self.episode_results else [])
            writer.writeheader()
            writer.writerows(self.episode_results)
        print(f"Policy saved to {os.path.abspath(policy_path)}")
        print(f"Training records saved to {os.path.abspath(self.output_dir)}")
        return self.agent


def main() -> None:
    """Parse training settings, optionally tune them, and export the final Q-table."""
    parser = argparse.ArgumentParser(description="Train a traffic signal controller")
    parser.add_argument("--agent_type", choices=["tabular", "neural"], default="tabular", help="Controller type")
    parser.add_argument("--episodes", type=int, default=200, help="Number of fresh training traffic schedules")
    parser.add_argument("--episode_length", type=int, default=3600, help="Training duration in seconds")
    parser.add_argument("--scenarios", nargs="+", default=["balanced", "ns_skew", "heavy", "spike"], help="Shared demand profiles used in rotation")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Tabular update fraction between zero and one")
    parser.add_argument("--epsilon_start", type=float, default=0.5, help="Initial chance of choosing a random legal action")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Exploration multiplier applied once after each episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount applied to the next legal state's value")
    parser.add_argument("--seed", type=int, default=1000, help="First training seed; evaluation reserves seeds 42 through 71")
    parser.add_argument("--policy_output", default=os.path.join("models", "tabular_policy.json"), help="Portable policy file used by the web runner")
    parser.add_argument("--output_root", default="output", help="Directory for training records")
    parser.add_argument("--print_interval", type=int, default=10, help="Episodes between progress lines")
    parser.add_argument("--use_ea", action="store_true", help="Run a two-generation evolutionary search before training")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.episode_length <= 3600 or args.print_interval < 1:
        parser.error("episodes and print_interval must be positive and episode_length must be at most 3600")
    unknown_scenarios = sorted(set(args.scenarios) - PRESET_PROFILES.keys())
    if unknown_scenarios:
        parser.error(f"unknown scenarios: {', '.join(unknown_scenarios)}")

    config = vars(args)
    if args.use_ea:
        config.update(evolutionary_search(args.scenarios, min(args.episode_length, 600), args.seed))
    trainer = EnhancedTrainingManager(config)
    trainer.train()


if __name__ == "__main__":
    main()
