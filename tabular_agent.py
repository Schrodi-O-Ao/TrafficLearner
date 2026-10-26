"""Seeded tabular Q-learning agent with legal-action masking and JSON checkpoints."""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np


class SimpleQLearningAgent:
    """Choose legal signal actions from learned state values and update their future return estimates."""

    def __init__(self, learning_rate: float = 0.1, epsilon_start: float = 0.5,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 gamma: float = 0.99, seed: int | None = None):
        if not 0 < learning_rate <= 1 or not 0 <= epsilon_min <= epsilon_start <= 1:
            raise ValueError("learning_rate and exploration values are outside their valid range")
        if not 0 < epsilon_decay <= 1 or not 0 <= gamma <= 1:
            raise ValueError("epsilon_decay and gamma must be between zero and one")
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple[int, ...], list[float]] = {}
        self.training_steps = 0
        self.episodes_completed = 0
        self.epsilon_history = [self.epsilon]

    def _initial_values(self, state: tuple[int, ...]) -> list[float]:
        """Give both legal actions the discounted demand-based estimate to avoid optimistic bias."""
        queue_bin_means = (0.0, 2.0, 5.5, 11.5, 20.0)
        expected_queue = sum(queue_bin_means[min(max(int(value), 0), 4)] for value in state[:4])
        scale = max(1.0 - self.gamma, 0.01)
        initial_value = -expected_queue / scale
        return [initial_value, initial_value]

    def _values(self, state: tuple[int, ...]) -> list[float]:
        """Return or initialize the two learned values for a discretized intersection state."""
        return self.q_table.setdefault(tuple(state), self._initial_values(state))

    def get_action(self, state: tuple, legal: Iterable[int] | None = None,
                   training: bool = True) -> int:
        """Choose an available action with epsilon-greedy exploration during training."""
        available_actions = tuple(int(action) for action in (legal if legal is not None else (0, 1)))
        if not available_actions or any(action not in (0, 1) for action in available_actions):
            raise ValueError("legal actions must contain zero, one, or both supported actions")
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))
        values = self._values(tuple(state))
        return max(available_actions, key=lambda action: values[action])

    def update(self, state: tuple, action: int, reward: float, next_state: tuple,
               done: bool = False, next_legal: Iterable[int] | None = None) -> None:
        """Update the selected action from its post-step reward and best legal next-state value."""
        if action not in (0, 1):
            raise ValueError("action must be zero or one")
        next_actions = tuple(next_legal if next_legal is not None else (0, 1))
        current_values = self._values(tuple(state))
        next_values = self._values(tuple(next_state))
        future_value = 0.0 if done or not next_actions else max(next_values[int(value)] for value in next_actions)
        target = float(reward) + self.gamma * future_value
        current_values[action] += self.learning_rate * (target - current_values[action])
        self.training_steps += 1

    def decay_exploration(self) -> None:
        """Reduce exploration once after a complete episode and record the value for the next one."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_completed += 1
        self.epsilon_history.append(self.epsilon)

    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration rate for compatibility with the optional hybrid trainer."""
        self.epsilon = float(np.clip(epsilon, self.epsilon_min, 1.0))

    def reset_episode(self) -> None:
        """Record a completed episode without changing the single configured exploration schedule."""
        self.episodes_completed += 1
        self.epsilon_history.append(self.epsilon)

    def get_statistics(self) -> dict:
        """Return table size, training counts, exploration rate, and learned value range."""
        all_values = [value for row in self.q_table.values() for value in row]
        return {
            "training_steps": self.training_steps,
            "episodes_completed": self.episodes_completed,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "minimum_q_value": min(all_values, default=0.0),
            "maximum_q_value": max(all_values, default=0.0),
        }

    def save(self, filepath: str) -> None:
        """Write learned state-action values and training settings as portable JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "format_version": 1,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "training_steps": self.training_steps,
            "episodes_completed": self.episodes_completed,
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in sorted(self.q_table.items())
            ],
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))

    def load(self, filepath: str) -> None:
        """Load a versioned JSON Q-table and restore its learned state-action values."""
        with open(filepath, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("format_version") != 1 or not isinstance(payload.get("q_table"), list):
            raise ValueError("unsupported tabular policy format")
        self.q_table = {
            tuple(int(value) for value in row["state"]): [float(value) for value in row["values"]]
            for row in payload["q_table"]
        }
        self.training_steps = int(payload.get("training_steps", 0))
        self.episodes_completed = int(payload.get("episodes_completed", 0))
        self.epsilon = float(payload.get("epsilon", 0.0))
