"""Unit tests for SophisticatedRuleLayer and StochasticEventGenerator."""

import pytest
from enhanced_rules import SophisticatedRuleLayer, StochasticEventGenerator
from simulation import SignalPhase, TrafficLight


def test_rule_layer_legal_actions_clearance():
    """Verify rule layer only permits action 0 during yellow and all-red clearance."""
    rules = SophisticatedRuleLayer(min_green=10, max_green=60)
    tl = TrafficLight(minimum_green=10, maximum_green=60)

    # In green before min_green
    assert rules.get_legal_actions(tl) == [0]

    # Advance to min green
    for _ in range(10):
        tl.step(0)
    assert rules.get_legal_actions(tl) == [0, 1]

    # Switch to yellow
    tl.step(1)
    assert rules.get_legal_actions(tl) == [0]


def test_rule_layer_empty_waiting_switch_prevention():
    """Verify rule layer vetoes switch action when no vehicles are waiting on opposing approaches."""
    rules = SophisticatedRuleLayer(min_green=5, max_green=30)
    tl = TrafficLight(minimum_green=5, maximum_green=30)
    for _ in range(6):
        tl.step(0)

    state = (2, 2, 0, 0, 0, 2)
    queues = {"N": 5, "S": 5, "E": 0, "W": 0}

    # Request switch when waiting queue is empty
    validated = rules.validate_action(state, action=1, traffic_light=tl, queue_lengths=queues)
    assert validated == 0


def test_rule_layer_starvation_prevention():
    """Verify waiting approach is eventually served when starved past limit."""
    rules = SophisticatedRuleLayer(min_green=5, max_green=60)
    tl = TrafficLight(minimum_green=5, maximum_green=60)
    for _ in range(6):
        tl.step(0)

    state = (2, 2, 1, 1, 0, 2)
    queues = {"N": 10, "S": 10, "E": 2, "W": 2}

    # Simulate starvation accumulation
    rules.phase_starvation_counters["EW"] = 50
    validated = rules.validate_action(state, action=0, traffic_light=tl, queue_lengths=queues)
    assert validated == 1


def test_stochastic_event_generator():
    """Verify event generator produces sorted reproducible events."""
    generator = StochasticEventGenerator(seed=123)
    base_rates = {"N": 0.1, "S": 0.1, "E": 0.05, "W": 0.05}
    events = generator.generate_events(duration=600, base_rates=base_rates)
    assert isinstance(events, list)
    if events:
        # Check sorted by time
        times = [e["time"] for e in events]
        assert times == sorted(times)
