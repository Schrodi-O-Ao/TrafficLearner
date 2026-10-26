"""Unit tests for paired evaluation runner and summary calculations."""

import os
import pytest
from evaluation import _legal_actions, _run_one, _summary, evaluate
from simulation import Intersection, SignalPhase


def test_legal_actions_bounds():
    """Verify legal actions during min green and max green limits."""
    sim = Intersection(seed=42, duration=100)
    # At start of green
    assert _legal_actions(sim.traffic_light) == [0]

    # Force time to min green
    sim.traffic_light.time_in_phase = 15
    assert _legal_actions(sim.traffic_light) == [0, 1]

    # Force time to max green
    sim.traffic_light.time_in_phase = 60
    assert _legal_actions(sim.traffic_light) == [1]


def test_summary_calculation():
    """Verify summary computes mean, std, min, and max."""
    values = [10.0, 20.0, 30.0]
    res = _summary(values)
    assert res["mean"] == 20.0
    assert res["minimum"] == 10.0
    assert res["maximum"] == 30.0
    assert res["sample_standard_deviation"] > 0


def test_evaluation_pipeline_smoke():
    """Verify evaluate runs across controllers and returns formatted dictionary."""
    policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tabular_policy.json")
    if not os.path.exists(policy_path):
        pytest.skip("Policy file not found")

    res = evaluate(scenarios=["balanced"], episode_length=60, runs=2, seed_start=42, policy_path=policy_path)
    assert "scenarios" in res
    assert "balanced" in res["scenarios"]
    controllers = res["scenarios"]["balanced"]["controllers"]
    assert "fixed_time" in controllers
    assert "queue_actuated" in controllers
    assert "q_learning" in controllers
