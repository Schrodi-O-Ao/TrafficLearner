"""Unit tests for utility discretization, index mapping, and helpers."""

import numpy as np
import pytest
import utilities


def test_discretize_queue_length():
    """Verify queue length bins map correctly."""
    assert utilities.discretize_queue_length(0) == 0
    assert utilities.discretize_queue_length(2) == 1
    assert utilities.discretize_queue_length(5) == 2
    assert utilities.discretize_queue_length(10) == 3
    assert utilities.discretize_queue_length(25) == 4


def test_discretize_time_in_phase():
    """Verify time in phase bins map correctly."""
    assert utilities.discretize_time_in_phase(3) == 0
    assert utilities.discretize_time_in_phase(10) == 1
    assert utilities.discretize_time_in_phase(30) == 2


def test_state_to_index_and_back():
    """Verify state encoding and decoding is bijective for all phase bins 0 through 3."""
    for phase in (0, 1, 2, 3):
        state = (1, 2, 0, 3, phase, 2)
        idx = utilities.state_to_index(state, phase_bins=4)
        recovered = utilities.index_to_state(idx, phase_bins=4)
        assert state == recovered


def test_generate_poisson_arrivals():
    """Verify Poisson arrivals generation constraints."""
    rng = np.random.default_rng(42)
    arrivals = utilities.generate_poisson_arrivals(rate=0.1, duration=100.0, rng=rng)
    assert isinstance(arrivals, list)
    if arrivals:
        assert min(arrivals) >= 0.0
        assert max(arrivals) <= 100.0
        assert arrivals == sorted(arrivals)


def test_format_time():
    """Verify seconds format into mm:ss strings."""
    assert utilities.format_time(0) == "00:00"
    assert utilities.format_time(65) == "01:05"
    assert utilities.format_time(3600) == "60:00"
