"""One four-way intersection with seeded arrivals, lane-aware discharge, and per-vehicle events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain

import numpy

import utilities
from scenario_profiles import PRESET_PROFILES, SIMULATION_DURATION, profile_to_dataset, sample_arrival_schedule
from utilities import DIRECTIONS

# The signal timer counts whole steps as seconds, so the step length is fixed rather than configurable.
TIME_STEP_SECONDS = 1.0


class SignalPhase(IntEnum):
    """Signal states; the integer values are sent to the browser and stored in controller state."""

    NORTH_SOUTH_GREEN = 0
    EAST_WEST_GREEN = 1
    YELLOW = 2
    ALL_RED = 3


GREEN_DIRECTIONS = {
    SignalPhase.NORTH_SOUTH_GREEN: ("N", "S"),
    SignalPhase.EAST_WEST_GREEN: ("E", "W"),
}


class TrafficLight:
    """Signal timer with minimum and maximum green limits and fixed yellow and all-red clearance."""

    def __init__(self, minimum_green: int = 10, maximum_green: int = 60,
                 yellow_duration: int = 3, all_red_duration: int = 1,
                 green_directions: dict[int, tuple[str, ...]] | None = None):
        if minimum_green < 1 or maximum_green < minimum_green or yellow_duration < 1 or all_red_duration < 1:
            raise ValueError("green limits must satisfy 1 <= minimum <= maximum and clearance times must be positive")
        self.minimum_green = int(minimum_green)
        self.maximum_green = int(maximum_green)
        self.yellow_duration = int(yellow_duration)
        self.all_red_duration = int(all_red_duration)
        self.green_directions_map = dict(green_directions or GREEN_DIRECTIONS)
        self.reset()

    def reset(self) -> None:
        """Return to the first green phase and clear the switch counter."""
        first_phase = next(iter(self.green_directions_map), SignalPhase.NORTH_SOUTH_GREEN)
        self.current_phase = first_phase
        self.time_in_phase = 0
        self.last_green_phase = first_phase
        self.switches_started = 0

    def is_green(self) -> bool:
        """Return whether a green phase is active, so a switch request can be considered."""
        return self.current_phase in self.green_directions_map

    def get_green_directions(self) -> tuple[str, ...]:
        """Return the approaches that may discharge now, which is empty during yellow and all-red."""
        return self.green_directions_map.get(self.current_phase, ())

    def can_depart(self, direction: str) -> bool:
        """Return whether vehicles on an approach currently have a green signal."""
        return direction in self.get_green_directions()

    def can_switch_next_step(self) -> bool:
        """Return whether a switch request made now would be accepted, since the timer advances before it is checked."""
        return self.is_green() and self.time_in_phase + 1 >= self.minimum_green

    def must_switch_next_step(self) -> bool:
        """Return whether the next step reaches maximum green and forces a change whatever is requested."""
        return self.is_green() and self.time_in_phase + 1 >= self.maximum_green

    def _enter(self, phase: SignalPhase) -> None:
        """Start a new phase with its timer at zero."""
        self.current_phase = phase
        self.time_in_phase = 0

    def step(self, switch_action: int = 0) -> bool:
        """Advance one second, start yellow on an accepted request or at maximum green, and report any phase change."""
        self.time_in_phase += 1
        if self.is_green():
            if (switch_action == 1 and self.time_in_phase >= self.minimum_green) or self.time_in_phase >= self.maximum_green:
                self.last_green_phase = self.current_phase
                self.switches_started += 1
                self._enter(SignalPhase.YELLOW)
                return True
        elif self.current_phase == SignalPhase.YELLOW:
            if self.time_in_phase >= self.yellow_duration:
                self._enter(SignalPhase.ALL_RED)
                return True
        elif self.time_in_phase >= self.all_red_duration:
            phases = list(self.green_directions_map.keys())
            if self.last_green_phase in phases:
                idx = phases.index(self.last_green_phase)
                next_phase = phases[(idx + 1) % len(phases)]
            else:
                next_phase = phases[0]
            self._enter(next_phase)
            return True
        return False


@dataclass(frozen=True)
class Vehicle:
    """One vehicle waiting on an approach, with the lane it joined and the second it arrived."""

    vehicle_id: int
    approach: str
    arrived_at: float
    lane_index: int = 0


class ApproachQueue:
    """First-in first-out queues for each lane of one approach."""

    def __init__(self, lane_count: int):
        self.lanes = [deque() for _ in range(lane_count)]

    def __len__(self) -> int:
        """Count every waiting vehicle on the approach across all of its lanes."""
        return sum(len(lane) for lane in self.lanes)

    def __iter__(self):
        """Iterate over waiting vehicles in arrival order, because vehicle identifiers rise with arrival time."""
        return iter(sorted(chain.from_iterable(self.lanes), key=lambda vehicle: vehicle.vehicle_id))

    def shortest_lane_index(self) -> int:
        """Return the lane with the fewest vehicles, preferring the lowest index on ties."""
        return min(range(len(self.lanes)), key=lambda index: len(self.lanes[index]))


class Intersection:
    """Advance queues, signal, and statistics one second at a time from a scheduled set of arrivals."""

    def __init__(self, arrival_rates: dict[str, float] | None = None, seed: int | None = None,
                 lanes: dict[str, int] | None = None, arrival_times: dict[str, list[float]] | None = None,
                 duration: float = SIMULATION_DURATION, saturation_flow: float = 0.5,
                 startup_lost_time: float = 2.0, minimum_green: int = 10, maximum_green: int = 60,
                 yellow_duration: int = 3, all_red_duration: int = 1,
                 approaches: tuple[str, ...] | None = None,
                 green_directions: dict[int, tuple[str, ...]] | None = None):
        if saturation_flow <= 0 or startup_lost_time < 0 or duration <= 0:
            raise ValueError("saturation_flow and duration must be positive and startup_lost_time cannot be negative")
        self.seed = seed
        self.duration = float(duration)
        self.saturation_flow = float(saturation_flow)
        self.startup_lost_time = float(startup_lost_time)
        self.time_step_seconds = TIME_STEP_SECONDS

        if approaches is not None:
            self.approaches = tuple(approaches)
        elif arrival_times is not None:
            self.approaches = tuple(sorted(arrival_times.keys()))
        elif arrival_rates is not None:
            self.approaches = tuple(sorted(arrival_rates.keys()))
        else:
            self.approaches = DIRECTIONS

        self.lanes = {direction: int((lanes or {}).get(direction, 1)) for direction in self.approaches}
        if any(not 1 <= count <= 4 for count in self.lanes.values()):
            raise ValueError("each approach must have between one and four lanes")
        self.arrival_rates = {direction: float((arrival_rates or {}).get(direction, 0.0)) for direction in self.approaches}
        self._fixed_arrivals = None if arrival_times is None else {
            direction: sorted(float(value) for value in arrival_times.get(direction, [])) for direction in self.approaches}
        self.traffic_light = TrafficLight(minimum_green, maximum_green, yellow_duration, all_red_duration,
                                          green_directions=green_directions)
        self.reset()

    def reset(self) -> None:
        """Restore the empty initial state and rebuild the arrival schedule so a reset run repeats exactly."""
        self.traffic_light.reset()
        self.random_generator = numpy.random.default_rng(self.seed)
        self.queues = {direction: ApproachQueue(self.lanes[direction]) for direction in self.approaches}
        self.arrival_times = self._fixed_arrivals or {
            direction: utilities.generate_poisson_arrivals(self.arrival_rates[direction], self.duration, self.random_generator)
            for direction in self.approaches}
        self.arrival_cursor = {direction: 0 for direction in self.approaches}
        self.current_time = 0.0
        self.next_vehicle_id = 0
        self.total_arrived = 0
        self.total_departed = 0
        self.total_wait_time = 0.0
        self.queue_seconds = 0.0
        self.departed_this_step = 0
        self.switch_started_this_step = False
        self.events_this_step: list[dict] = []
        self._service_credit = {direction: [0.0] * self.lanes[direction] for direction in self.approaches}
        self._green_elapsed = {direction: 0.0 for direction in self.approaches}

    def queue_lengths(self) -> dict[str, int]:
        """Return the number of vehicles waiting on each approach."""
        return {direction: len(queue) for direction, queue in self.queues.items()}

    def _add_arrivals(self, end_time: float) -> float:
        """Queue every scheduled vehicle that arrives before end_time and return the waiting seconds they add this step."""
        added_wait_seconds = 0.0
        for direction in self.approaches:
            schedule, cursor = self.arrival_times[direction], self.arrival_cursor[direction]
            queue = self.queues[direction]
            while cursor < len(schedule) and schedule[cursor] < end_time:
                lane_index = queue.shortest_lane_index()
                vehicle = Vehicle(self.next_vehicle_id, direction, schedule[cursor], lane_index)
                queue.lanes[lane_index].append(vehicle)
                self.next_vehicle_id += 1
                self.total_arrived += 1
                added_wait_seconds += end_time - vehicle.arrived_at
                self.events_this_step.append({"type": "vehicle_entered", "t": vehicle.arrived_at,
                                              "vehicle_id": vehicle.vehicle_id, "direction": direction.lower(),
                                              "lane_index": lane_index})
                cursor += 1
            self.arrival_cursor[direction] = cursor
        return added_wait_seconds

    def _discharge(self, end_time: float) -> int:
        """Release vehicles from green approaches at saturation flow per lane after the start-up delay."""
        departed = 0
        green_directions = self.traffic_light.get_green_directions()
        for direction in self.approaches:
            if direction not in green_directions:
                # Capacity cannot be saved through red, and the next green starts with its start-up loss again.
                self._green_elapsed[direction] = 0.0
                self._service_credit[direction] = [0.0] * self.lanes[direction]
                continue
            self._green_elapsed[direction] += self.time_step_seconds
            if self._green_elapsed[direction] <= self.startup_lost_time:
                continue
            for lane_index, lane in enumerate(self.queues[direction].lanes):
                credit = self._service_credit[direction][lane_index] + self.saturation_flow * self.time_step_seconds
                while credit >= 1.0 and lane:
                    vehicle = lane.popleft()
                    credit -= 1.0
                    wait = end_time - vehicle.arrived_at
                    self.total_wait_time += wait
                    departed += 1
                    self.events_this_step.append({"type": "vehicle_exited", "t": end_time,
                                                  "vehicle_id": vehicle.vehicle_id, "direction": direction.lower(),
                                                  "lane_index": lane_index, "arrival_time": vehicle.arrived_at,
                                                  "wait_time": wait})
                # Unused capacity is dropped, except one vehicle's worth so a lone arrival passes without waiting.
                self._service_credit[direction][lane_index] = credit if lane else min(credit, 1.0)
        return departed

    def step(self, switch_action: int = 0) -> tuple[dict[str, int], int, float]:
        """Advance one second and return queue lengths, vehicles served, and the reward of negative waiting vehicles."""
        start_time = self.current_time
        end_time = start_time + self.time_step_seconds
        self.events_this_step = []
        waiting_at_start = sum(len(queue) for queue in self.queues.values())
        arrival_wait_seconds = self._add_arrivals(end_time)
        self.departed_this_step = self._discharge(end_time)
        self.total_departed += self.departed_this_step
        # Vehicles present at the start wait the full second; arrivals wait from their timestamp to end_time.
        self.queue_seconds += waiting_at_start * self.time_step_seconds + arrival_wait_seconds

        previous_phase = self.traffic_light.current_phase
        switches_before = self.traffic_light.switches_started
        if self.traffic_light.step(switch_action):
            self.events_this_step.append({"type": "signal_changed", "t": end_time,
                                          "from_phase": int(previous_phase),
                                          "to_phase": int(self.traffic_light.current_phase)})
        self.switch_started_this_step = self.traffic_light.switches_started > switches_before
        self.current_time = end_time

        queue_lengths = self.queue_lengths()
        reward = -float(sum(queue_lengths.values())) * self.time_step_seconds
        return queue_lengths, self.departed_this_step, reward

    def get_state(self) -> tuple[int, int, int, int, int, int]:
        """Return four queue bins, a phase bin that separates clearance direction, and the time-in-phase bin."""
        light = self.traffic_light
        if light.current_phase == SignalPhase.NORTH_SOUTH_GREEN:
            phase_bin = 0
        elif light.current_phase == SignalPhase.EAST_WEST_GREEN:
            phase_bin = 1
        else:
            phase_bin = 2 if light.last_green_phase == SignalPhase.NORTH_SOUTH_GREEN else 3
        queue_bins = tuple(
            utilities.discretize_queue_length(len(self.queues[direction])) if direction in self.queues else 0
            for direction in DIRECTIONS
        )
        return (*queue_bins, phase_bin, utilities.discretize_time_in_phase(light.time_in_phase))

    def get_statistics(self) -> dict:
        """Report waits, queue length, throughput, backlog, and signal switches measured so far."""
        queue_lengths = self.queue_lengths()
        elapsed = max(self.current_time, self.time_step_seconds)
        return {
            "time": self.current_time,
            "queue_lengths": queue_lengths,
            "total_queue": sum(queue_lengths.values()),
            "total_arrived": self.total_arrived,
            "total_departed": self.total_departed,
            "average_wait_completed": self.total_wait_time / max(self.total_departed, 1),
            "average_wait_experienced": self.queue_seconds / max(self.total_arrived, 1),
            "average_queue_length": self.queue_seconds / elapsed,
            "throughput_per_minute": self.total_departed / (elapsed / 60.0),
            "signal_switches": self.traffic_light.switches_started,
            "phase": int(self.traffic_light.current_phase),
            "time_in_phase": self.traffic_light.time_in_phase,
        }


def create_scenario(scenario_name: str, seed: int | None = None, duration: float = SIMULATION_DURATION,
                    arrival_times: dict[str, list[float]] | None = None) -> Intersection:
    """Build an intersection for a shared preset, sampling arrivals from the seed unless a schedule is supplied."""
    if scenario_name not in PRESET_PROFILES:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    arrivals = arrival_times or sample_arrival_schedule(profile_to_dataset(PRESET_PROFILES[scenario_name]), seed, duration)
    return Intersection(arrival_times=arrivals, seed=seed, duration=duration)


if __name__ == "__main__":
    demonstration = create_scenario("heavy", seed=42, duration=300)
    for second in range(300):
        demonstration.step(0)
        if second % 60 == 59:
            snapshot = demonstration.get_statistics()
            print(f"t={snapshot['time']:.0f}s queues={snapshot['queue_lengths']} served={snapshot['total_departed']}")
    print({key: round(value, 3) if isinstance(value, float) else value for key, value in demonstration.get_statistics().items()})
