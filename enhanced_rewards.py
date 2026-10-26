import numpy as np
from typing import Dict, List, Tuple


class EnhancedRewardCalculator:
    """Advanced reward shaping for traffic light control."""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Reward configuration parameters
        """
        self.config = config or {
            'w_wait': 1.0,           # Base waiting penalty
            'w_switch': 0.2,         # Base switching penalty
            'w_throughput': 0.1,     # Throughput reward
            'dynamic_switch': True,   # Dynamic switching penalty
            'direction_weighted': True,  # Direction-weighted penalties
        }
        
        self.episode_reward = 0.0
        self.total_departed = 0
        self.total_switches = 0
    
    def calculate_reward(self, queue_lengths: Dict[str, int], departed: int, 
                        switch_action: int, phase: int, 
                        total_vehicles_served: Dict[str, int] = None) -> float:
        """
        Calculate normalized reward to prevent value explosion.
        
        Args:
            queue_lengths: Current queue lengths per direction
            departed: Number of vehicles departed in this step
            switch_action: Action taken (0=keep, 1=switch)
            phase: Current traffic phase
            total_vehicles_served: Total vehicles served per direction (optional)
        
        Returns:
            Normalized reward value
        """
        # Normalization constants
        WAIT_NORM = 60.0  # Normalize wait to minutes
        QUEUE_NORM = 50.0  # Normalize queue to reasonable scale
        
        # 1. Normalized waiting penalty (minutes of delay)
        total_waiting = sum(queue_lengths.values())
        wait_penalty = -self.config['w_wait'] * (total_waiting / WAIT_NORM)
        
        # 2. Normalized max queue penalty (prevents explosion)
        max_queue = max(queue_lengths.values()) if queue_lengths else 0
        max_queue_penalty = -self.config.get('w_maxq', 0.2) * (max_queue / QUEUE_NORM)
        
        # 3. Switch penalty (small, fixed)
        switch_penalty = -self.config['w_switch'] if switch_action == 1 else 0
        
        # 4. Normalized throughput reward (vehicles per arrival)
        total_arrived = sum(total_vehicles_served.values()) if total_vehicles_served else 0
        if total_arrived > 0:
            throughput_reward = self.config['w_throughput'] * (departed / min(total_arrived, 10))
        elif departed > 0:
            throughput_reward = self.config['w_throughput'] * departed
        else:
            throughput_reward = 0.0
        
        # Combine all components (removed variance and worst_queue penalties)
        total_reward = wait_penalty + max_queue_penalty + switch_penalty + throughput_reward
        
        # Update tracking
        self.episode_reward += total_reward
        self.total_departed += departed
        self.total_switches += 1 if switch_action == 1 else 0
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific tracking counters."""
        self.episode_reward = 0.0
        self.total_departed = 0
        self.total_switches = 0


class AdaptiveLearningScheduler:
    """Adaptive learning rate and exploration scheduling."""
    
    def __init__(self, initial_alpha: float = 0.1, initial_epsilon: float = 0.5,
                 min_alpha: float = 0.01, min_epsilon: float = 0.05):
        """
        Args:
            initial_alpha: Initial learning rate
            initial_epsilon: Initial exploration rate
            min_alpha: Minimum learning rate
            min_epsilon: Minimum exploration rate
        """
        self.initial_alpha = initial_alpha
        self.initial_epsilon = initial_epsilon
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        
        self.current_alpha = initial_alpha
        self.current_epsilon = initial_epsilon
        
        # Track performance for adaptive scheduling
        self.reward_history = []
        self.performance_window = 100
        
    def update_parameters(self, episode: int, total_episodes: int, 
                         recent_reward: float):
        """
        Update learning parameters based on progress.
        
        Args:
            episode: Current episode number
            total_episodes: Total number of episodes
            recent_reward: Recent average reward
        """
        self.reward_history.append(recent_reward)
        
        # Keep only recent history
        if len(self.reward_history) > self.performance_window:
            self.reward_history = self.reward_history[-self.performance_window:]
        
        # Adaptive epsilon decay
        if len(self.reward_history) >= 20:
            recent_avg = np.mean(self.reward_history[-10:])
            older_avg = np.mean(self.reward_history[-20:-10])
            
            # If learning is stagnating, slow epsilon decay
            if abs(recent_avg - older_avg) < 0.1:
                epsilon_decay = 0.999  # Slower decay
            else:
                epsilon_decay = 0.995  # Normal decay
            
            self.current_epsilon = max(
                self.min_epsilon,
                self.current_epsilon * epsilon_decay
            )
        else:
            # Standard decay for early episodes
            progress = episode / total_episodes
            self.current_epsilon = max(
                self.min_epsilon,
                self.initial_epsilon * (1 - progress) ** 2
            )
        
        # Adaptive alpha decay
        if episode > total_episodes * 0.7:  # Start decaying alpha after 70% progress
            alpha_decay = 0.999
            self.current_alpha = max(
                self.min_alpha,
                self.current_alpha * alpha_decay
            )
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get current learning parameters."""
        return self.current_alpha, self.current_epsilon


class EnhancedStateRepresentation:
    """Enhanced state representation with temporal context and ratios."""
    
    def __init__(self, history_length: int = 5):
        """
        Args:
            history_length: Number of previous steps to include in state
        """
        self.history_length = history_length
        self.queue_history = []
        self.action_history = []
        self.phase_history = []
    
    def update_history(self, queue_lengths: Dict[str, int], 
                      action: int, phase: int):
        """Update history with current step."""
        self.queue_history.append(queue_lengths.copy())
        self.action_history.append(action)
        self.phase_history.append(phase)
        
        # Keep only recent history
        if len(self.queue_history) > self.history_length:
            self.queue_history = self.queue_history[-self.history_length:]
            self.action_history = self.action_history[-self.history_length:]
            self.phase_history = self.phase_history[-self.history_length:]
    
    def get_enhanced_state(self, current_queues: Dict[str, int], 
                          phase: int, time_in_phase: int) -> List[float]:
        """
        Get enhanced state representation.
        
        Args:
            current_queues: Current queue lengths
            phase: Current phase
            time_in_phase: Time in current phase
        
        Returns:
            Enhanced state vector
        """
        state_features = []
        
        # Current queue ratios (normalized)
        total_queue = sum(current_queues.values()) + 1  # +1 to avoid division by zero
        for direction in ['N', 'S', 'E', 'W']:
            ratio = current_queues.get(direction, 0) / total_queue
            state_features.append(ratio)

        # Absolute queue sizes distinguish equally shaped light and heavy demand.
        state_features.extend(min(max(current_queues.get(direction, 0), 0), 100) / 100.0
                              for direction in ['N', 'S', 'E', 'W'])
        
        # Phase and time
        state_features.append(phase / 1.0)  # Normalize phase
        state_features.append(time_in_phase / 60.0)  # Normalize time
        
        # Pressure metrics
        ns_pressure = current_queues.get('N', 0) + current_queues.get('S', 0)
        ew_pressure = current_queues.get('E', 0) + current_queues.get('W', 0)
        pressure_diff = (ns_pressure - ew_pressure) / (total_queue + 1)
        state_features.append(pressure_diff)
        
        # Temporal trends (if history available)
        if len(self.queue_history) >= 3:
            # Queue trend (increasing/decreasing)
            recent_queues = [sum(q.values()) for q in self.queue_history[-3:]]
            if len(recent_queues) >= 2:
                trend = (recent_queues[-1] - recent_queues[0]) / (len(recent_queues) * total_queue + 1)
                state_features.append(trend)
            else:
                state_features.append(0.0)
        else:
            state_features.append(0.0)
        
        # Recent action frequency
        if len(self.action_history) >= 5:
            recent_switches = sum(self.action_history[-5:])
            switch_frequency = recent_switches / 5.0
            state_features.append(switch_frequency)
        else:
            state_features.append(0.0)
        
        return state_features
    
    def reset_episode(self):
        """Reset episode-specific history."""
        self.queue_history.clear()
        self.action_history.clear()
        self.phase_history.clear()


if __name__ == "__main__":
    # Test enhanced components
    print("Testing enhanced reward calculator...")
    reward_calc = EnhancedRewardCalculator()
    
    test_queues = {'N': 5, 'S': 3, 'E': 8, 'W': 2}
    test_served = {'N': 10, 'S': 8, 'E': 12, 'W': 6}
    
    reward = reward_calc.calculate_reward(test_queues, departed=2, switch_action=1, 
                                        phase=0, total_vehicles_served=test_served)
    print(f"Enhanced reward: {reward:.3f}")
    
    print("\nTesting adaptive scheduler...")
    scheduler = AdaptiveLearningScheduler()
    
    for episode in range(10):
        scheduler.update_parameters(episode, 100, recent_reward=-50)
        alpha, epsilon = scheduler.get_parameters()
        print(f"Episode {episode}: alpha={alpha:.3f}, epsilon={epsilon:.3f}")
    
    print("\nTesting enhanced state representation...")
    state_rep = EnhancedStateRepresentation()
    
    for step in range(5):
        state_rep.update_history(test_queues, action=step%2, phase=step%2)
        enhanced_state = state_rep.get_enhanced_state(test_queues, phase=step%2, time_in_phase=step*5)
        print(f"Step {step}: Enhanced state length={len(enhanced_state)}")
    
    print("Enhanced components test completed!")
