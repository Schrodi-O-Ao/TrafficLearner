"""Unit tests for TrafficAI traffic light and intersection simulation."""

import pytest
from simulation import Intersection, SignalPhase, TrafficLight, create_scenario
from utilities import DIRECTIONS


def test_traffic_light_initial_state():
    """Verify traffic light initializes to north-south green with zero elapsed time."""
    tl = TrafficLight(minimum_green=10, maximum_green=60, yellow_duration=3, all_red_duration=1)
    assert tl.current_phase == SignalPhase.NORTH_SOUTH_GREEN
    assert tl.time_in_phase == 0
    assert tl.is_green()
    assert tl.get_green_directions() == ("N", "S")


def test_traffic_light_min_green_guard():
    """Verify traffic light does not switch before minimum green duration has elapsed."""
    tl = TrafficLight(minimum_green=10, maximum_green=60, yellow_duration=3, all_red_duration=1)
    for _ in range(5):
        changed = tl.step(switch_action=1)
        assert not changed
        assert tl.current_phase == SignalPhase.NORTH_SOUTH_GREEN

    # Advance to step 9 (time_in_phase reaches 10 upon step)
    for _ in range(4):
        tl.step(switch_action=0)
    
    # At step 10, a switch action should be accepted
    changed = tl.step(switch_action=1)
    assert changed
    assert tl.current_phase == SignalPhase.YELLOW


def test_traffic_light_phase_cycle():
    """Verify full signal cycle through yellow, all-red, and opposing green."""
    tl = TrafficLight(minimum_green=5, maximum_green=20, yellow_duration=2, all_red_duration=1)
    for _ in range(5):
        tl.step(0)
    tl.step(1)
    assert tl.current_phase == SignalPhase.YELLOW

    tl.step(0)
    tl.step(0)
    assert tl.current_phase == SignalPhase.ALL_RED

    tl.step(0)
    assert tl.current_phase == SignalPhase.EAST_WEST_GREEN
    assert tl.get_green_directions() == ("E", "W")


def test_intersection_reset_repeatability():
    """Verify identical seeds produce identical arrivals and metrics upon reset."""
    rates = {"N": 0.1, "S": 0.1, "E": 0.05, "W": 0.05}
    sim = Intersection(arrival_rates=rates, seed=123, duration=100)
    for _ in range(50):
        sim.step(0)
    stats1 = sim.get_statistics()

    sim.reset()
    for _ in range(50):
        sim.step(0)
    stats2 = sim.get_statistics()

    assert stats1["total_arrived"] == stats2["total_arrived"]
    assert stats1["total_departed"] == stats2["total_departed"]
    assert stats1["total_queue"] == stats2["total_queue"]


def test_intersection_custom_approaches_t_junction():
    """Verify intersection supports 3-way T-junction topologies without errors."""
    approaches = ("N", "S", "E")
    rates = {"N": 0.08, "S": 0.08, "E": 0.10}
    lanes = {"N": 1, "S": 1, "E": 1}
    sim = Intersection(arrival_rates=rates, seed=42, lanes=lanes, approaches=approaches, duration=120)

    assert sim.approaches == ("N", "S", "E")
    assert "W" not in sim.queues
    assert len(sim.queues) == 3

    for _ in range(30):
        queues, departed, reward = sim.step(0)
        assert isinstance(queues, dict)
        assert isinstance(departed, int)
        assert isinstance(reward, float)

    state = sim.get_state()
    assert len(state) == 6
    assert state[3] == 0  # Missing W approach maps to 0 queue bin


def test_intersection_statistics_zero_elapsed():
    """Verify get_statistics does not raise ZeroDivisionError at t=0."""
    sim = Intersection(seed=42, duration=10)
    stats = sim.get_statistics()
    assert stats["time"] == 0.0
    assert stats["average_wait_completed"] == 0.0
    assert stats["average_wait_experienced"] == 0.0
    assert stats["throughput_per_minute"] == 0.0
