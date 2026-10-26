"""Unit tests for baseline controllers and tabular Q-learning agent."""

import os
import tempfile
import pytest
from baseline_controller import FixedTimeController, QueueActuatedController
from simulation import Intersection, TrafficLight
from tabular_agent import SimpleQLearningAgent


def test_fixed_time_controller():
    """Verify fixed-time controller requests switches at target green time."""
    tl = TrafficLight(minimum_green=5, maximum_green=30)
    controller = FixedTimeController(green_time=10)

    # Before green target
    for _ in range(8):
        tl.step(0)
        assert controller.get_action(tl) == 0

    # At step 9 (next step will hit 10)
    tl.step(0)
    assert controller.get_action(tl) == 1


def test_queue_actuated_controller():
    """Verify queue-actuated controller switches when waiting queue exceeds threshold."""
    tl = TrafficLight(minimum_green=5, maximum_green=30)
    controller = QueueActuatedController(min_green=5, max_green=30, threshold=3)

    # Before min green
    for _ in range(3):
        tl.step(0)
    queues = {"N": 1, "S": 1, "E": 10, "W": 10}
    assert controller.get_action(tl, queues) == 0

    # Past min green with waiting > current + threshold
    for _ in range(3):
        tl.step(0)
    assert controller.get_action(tl, queues) == 1

    # Waiting queue less than current queue
    balanced_queues = {"N": 5, "S": 5, "E": 2, "W": 2}
    assert controller.get_action(tl, balanced_queues) == 0


def test_queue_actuated_controller_t_junction():
    """Verify queue-actuated controller handles non-4-way queue dictionaries."""
    tl = TrafficLight(minimum_green=5, maximum_green=30)
    for _ in range(6):
        tl.step(0)
    controller = QueueActuatedController(min_green=5, max_green=30, threshold=2)

    # Only N, S, E present
    t_queues = {"N": 0, "S": 1, "E": 6}
    action = controller.get_action(tl, t_queues)
    assert action == 1


def test_tabular_agent_action_and_update():
    """Verify Q-learning agent chooses legal actions and updates state values."""
    agent = SimpleQLearningAgent(learning_rate=0.1, epsilon_start=0.0, epsilon_min=0.0, seed=42)
    state = (0, 0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0, 1)

    # Greedy action on legal subset
    action = agent.get_action(state, legal=[0, 1], training=False)
    assert action in (0, 1)

    # Action when only action 0 is legal
    restricted_action = agent.get_action(state, legal=[0], training=False)
    assert restricted_action == 0

    # Update value
    agent.update(state, action=0, reward=-5.0, next_state=next_state, done=False, next_legal=[0, 1])
    assert agent.training_steps == 1
    stats = agent.get_statistics()
    assert stats["q_table_size"] >= 1


def test_tabular_agent_save_load_round_trip():
    """Verify saving and loading policy preserves Q-values and metadata."""
    agent = SimpleQLearningAgent(learning_rate=0.1, epsilon_start=0.2, seed=42)
    state = (2, 1, 3, 0, 1, 2)
    agent.update(state, action=1, reward=-12.5, next_state=state, done=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "policy.json")
        agent.save(path)

        loaded = SimpleQLearningAgent(seed=0)
        loaded.load(path)

        assert loaded.training_steps == agent.training_steps
        assert loaded.q_table[state] == agent.q_table[state]
