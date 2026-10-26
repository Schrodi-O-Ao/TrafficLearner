"""Fixed-time and queue-actuated controls for comparing signal timing policies."""


class FixedTimeController:
    """Request a signal change after the configured green interval has elapsed."""

    def __init__(self, green_time: int = 15, yellow_time: int = 3, all_red_time: int = 1):
        if green_time < 1 or yellow_time < 1 or all_red_time < 0:
            raise ValueError("green and yellow durations must be positive and all-red cannot be negative")
        self.green_time = int(green_time)
        self.yellow_time = int(yellow_time)
        self.all_red_time = int(all_red_time)

    def get_action(self, traffic_light) -> int:
        """Request a change only during green and on the step that reaches the target green time."""
        if traffic_light.current_phase not in (0, 1):
            return 0
        return int(traffic_light.time_in_phase + 1 >= self.green_time)

    def reset(self) -> None:
        """Keep no episode state because the traffic light owns elapsed phase time."""
        return None


class QueueActuatedController:
    """Switch after minimum green when the waiting opposite phase has a larger queue."""

    def __init__(self, min_green: int = 10, max_green: int = 60, threshold: int = 3):
        if min_green < 1 or max_green < min_green or threshold < 0:
            raise ValueError("actuated timing and queue threshold are invalid")
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.threshold = int(threshold)

    def get_action(self, traffic_light, queues: dict[str, int]) -> int:
        """Request a legal green change at the queue threshold or maximum green limit."""
        if traffic_light.current_phase not in (0, 1):
            return 0
        next_green_age = traffic_light.time_in_phase + 1
        if next_green_age < self.min_green:
            return 0
        if next_green_age >= self.max_green:
            return 1
        current_directions = traffic_light.get_green_directions()
        current_queue = sum(queues.get(direction, 0) for direction in current_directions)
        waiting_queue = sum(queues.get(direction, 0) for direction in queues if direction not in current_directions)
        return int(waiting_queue > current_queue + self.threshold)

    def reset(self) -> None:
        """Keep no episode state because queue counts and phase time come from the simulator."""
        return None
