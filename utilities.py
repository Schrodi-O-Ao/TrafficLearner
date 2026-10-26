import numpy as np
from typing import Tuple, List

DIRECTIONS = ("N", "S", "E", "W")


def discretize_queue_length(queue_length: int, bins: List[Tuple[int, int]] = None) -> int:
    if bins is None:
        bins = [(0, 0), (1, 3), (4, 7), (8, 15), (16, float('inf'))]
    
    for i, (min_val, max_val) in enumerate(bins):
        if min_val <= queue_length <= max_val:
            return i
    return len(bins) - 1


def discretize_time_in_phase(time_in_phase: int, bins: List[Tuple[int, int]] = None) -> int:
    if bins is None:
        bins = [(0, 5), (6, 15), (16, float('inf'))]
    
    for i, (min_val, max_val) in enumerate(bins):
        if min_val <= time_in_phase <= max_val:
            return i
    return len(bins) - 1


def state_to_index(state: Tuple[int, int, int, int, int, int], 
                   queue_bins: int = 5, phase_bins: int = 4, time_bins: int = 3) -> int:
    """Encode a discretized state tuple into a single integer index."""
    queue_N, queue_S, queue_E, queue_W, phase, time_in_phase = state
    
    index = 0
    index += queue_N * (queue_bins ** 3) * phase_bins * time_bins
    index += queue_S * (queue_bins ** 2) * phase_bins * time_bins
    index += queue_E * queue_bins * phase_bins * time_bins
    index += queue_W * phase_bins * time_bins
    index += phase * time_bins
    index += time_in_phase
    
    return index


def index_to_state(index: int, queue_bins: int = 5, phase_bins: int = 4, time_bins: int = 3) -> Tuple[int, int, int, int, int, int]:
    """Decode an integer index back into a discretized state tuple."""
    time_in_phase = index % time_bins
    index //= time_bins
    
    phase = index % phase_bins
    index //= phase_bins
    
    queue_W = index % queue_bins
    index //= queue_bins
    
    queue_E = index % queue_bins
    index //= queue_bins
    
    queue_S = index % queue_bins
    index //= queue_bins
    
    queue_N = index % queue_bins
    
    return (queue_N, queue_S, queue_E, queue_W, phase, time_in_phase)


def create_rng(seed: int) -> np.random.Generator:
    """Create an independent NumPy random generator so separate simulations do not share draws."""
    return np.random.default_rng(seed)


def generate_poisson_arrivals(
    rate: float,
    duration: float,
    rng: np.random.Generator,
) -> list[float]:
    if rng is None or not hasattr(rng, "exponential"):
        raise ValueError("pass a numpy.random.Generator")

    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be a finite, non-negative number")

    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be a finite, non-negative number")

    if rate == 0 or duration == 0:
        return []

    arrivals = []
    current_time = 0.0

    while True:
        gap_seconds = rng.exponential(scale=1.0 / rate)
        current_time += gap_seconds

        if current_time >= duration:
            break

        arrivals.append(current_time)

    return arrivals


def calculate_wait_times(arrival_times: List[int], departure_times: List[int]) -> List[float]:
    if len(arrival_times) != len(departure_times):
        raise ValueError("Arrival and departure lists must have same length")
    
    return [dep - arr for arr, dep in zip(arrival_times, departure_times)]


def moving_average(data: List[float], window_size: int) -> List[float]:
    if window_size <= 0:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        result.append(sum(window) / len(window))
    
    return result


def format_time(seconds: int) -> str:
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def create_queue_bar(length: int, max_length: int = 20) -> str:
    if length == 0:
        return "."
    
    bar_length = min(length, max_length)
    return "#" * bar_length


if __name__ == "__main__":
    # Test discretization functions
    print("Testing discretization functions:")
    
    # Test queue discretization
    test_queues = [0, 2, 5, 10, 20]
    for q in test_queues:
        bin_idx = discretize_queue_length(q)
        print(f"Queue {q} -> bin {bin_idx}")
    
    # Test time discretization
    test_times = [3, 8, 20]
    for t in test_times:
        bin_idx = discretize_time_in_phase(t)
        print(f"Time {t} -> bin {bin_idx}")
    
    # Test state conversion
    state = (1, 2, 0, 3, 1, 2)
    index = state_to_index(state)
    recovered_state = index_to_state(index)
    print(f"State {state} -> index {index} -> recovered {recovered_state}")
    print(f"Conversion correct: {state == recovered_state}")
