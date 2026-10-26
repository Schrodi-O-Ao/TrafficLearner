from __future__ import annotations

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from collections import deque
import enhanced_rewards
from tabular_agent import SimpleQLearningAgent

# Try to import torch, make it optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network features disabled. Install with: pip install torch")


class NeuralNetworkAgent:
    """Deep Q-Network agent for traffic light control."""
    
    def __init__(self, state_size: int = 13, action_size: int = 2,
                 hidden_size: int = 64, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon_start: float = 0.5,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 32,
                 target_update_interval: int = 500, seed: int | None = None):
        """
        Args:
            state_size: Size of state representation
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Experience replay buffer size
            batch_size: Training batch size
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralNetworkAgent. Install with: pip install torch")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = max(1, int(target_update_interval))
        self.rng = np.random.default_rng(seed)
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network(state_size, hidden_size, action_size).to(self.device)
        self.target_network = self._build_network(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_steps = 0
        self.episodes_completed = 0
        self.epsilon = epsilon_start
        self.epsilon_history = [epsilon_start]
        self.loss_history = []
        self.reward_history = []
        
        # Update target network
        self.update_target_network()
    
    def _build_network(self, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
        """Build neural network architecture."""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def get_action(self, state: List[float], training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state representation
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        
        was_training = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = int(self.q_network(state_tensor).argmax().item())
        if was_training:
            self.q_network.train()
        return action
    
    def remember(self, state: List[float], action: int, reward: float,
                next_state: List[float], done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        self.q_network.train()
        self.target_network.eval()
        batch = self.rng.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update statistics
        self.training_steps += 1
        self.loss_history.append(loss.item())
        if self.training_steps % self.target_update_interval == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.episodes_completed += 1
        self.epsilon_history.append(self.epsilon)
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        recent_rewards = self.reward_history[-100:] if len(self.reward_history) >= 100 else self.reward_history
        recent_losses = self.loss_history[-100:] if len(self.loss_history) >= 100 else self.loss_history
        
        return {
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
            'current_epsilon': self.epsilon,
            'avg_reward_recent': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_loss_recent': np.mean(recent_losses) if recent_losses else 0,
            'buffer_size': len(self.memory),
            'device': str(self.device)
        }
    
    def save(self, filepath: str):
        """Save agent state to file."""
        agent_data = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
            'loss_history': self.loss_history[-1000:],
            'reward_history': self.reward_history[-1000:],
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(agent_data, filepath)
    
    def load(self, filepath: str):
        """Load agent state from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent file not found: {filepath}")
        
        agent_data = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(agent_data['q_network_state'])
        self.target_network.load_state_dict(agent_data['target_network_state'])
        self.optimizer.load_state_dict(agent_data['optimizer_state'])
        self.epsilon = agent_data['epsilon']
        self.training_steps = agent_data['training_steps']
        self.episodes_completed = agent_data['episodes_completed']
        
        if 'loss_history' in agent_data:
            self.loss_history = agent_data['loss_history']
        if 'reward_history' in agent_data:
            self.reward_history = agent_data['reward_history']


# Simple tabular Q-learning agent (since agent_tabular.py was deleted)
class EnhancedHybridAgent:
    """Hybrid agent combining enhanced RL with sophisticated rules."""
    
    def __init__(self, agent_type: str = 'neural', config: Dict = None):
        """
        Args:
            agent_type: Type of RL agent ('tabular' or 'neural')
            config: Configuration parameters
        """
        self.agent_type = agent_type
        self.config = config or {}
        
        # Check if neural network is available
        if agent_type == 'neural' and not TORCH_AVAILABLE:
            print("PyTorch not available, falling back to tabular agent")
            self.agent_type = 'tabular'
        
        # Initialize components
        if self.agent_type == 'neural':
            self.rl_agent = NeuralNetworkAgent(
                state_size=self.config.get('state_size', 13),
                learning_rate=self.config.get('learning_rate', 0.001),
                epsilon_start=self.config.get('epsilon_start', 0.5)
            )
            self.state_rep = enhanced_rewards.EnhancedStateRepresentation(
                history_length=self.config.get('history_length', 5)
            )
        else:  # tabular
            self.rl_agent = SimpleQLearningAgent(
                learning_rate=self.config.get('learning_rate', 0.1),
                epsilon_start=self.config.get('epsilon_start', 0.5)
            )
            self.state_rep = None
        
        self._last_enhanced_state = None
        
        # Enhanced components
        self.reward_calculator = enhanced_rewards.EnhancedRewardCalculator(
            self.config.get('reward_config', {})
        )
        self.learning_scheduler = enhanced_rewards.AdaptiveLearningScheduler(
            initial_alpha=self.config.get('learning_rate', 0.1),
            initial_epsilon=self.config.get('epsilon_start', 0.5)
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_wait_times = []
        self.episode_queue_lengths = []
        self.enhanced_rewards_history = []
    
    def get_action(self, state: Tuple[int, int, int, int, int, int],
                   queue_lengths: Dict[str, int], phase: int, 
                   time_in_phase: int, training: bool = True) -> int:
        if self.agent_type == 'neural' and self.state_rep:
            enhanced_state = self.state_rep.get_enhanced_state(
                queue_lengths, phase, time_in_phase
            )
            self._last_enhanced_state = enhanced_state
            action = self.rl_agent.get_action(enhanced_state, training=training)
        else:
            self._last_enhanced_state = None
            action = self.rl_agent.get_action(state, training=training)
        
        return action
    
    def update(self, state: Tuple[int, int, int, int, int, int],
               action: int, reward: float, 
               next_state: Tuple[int, int, int, int, int, int],
               queue_lengths: Dict[str, int], phase: int,
               time_in_phase: int, done: bool = False,
               next_queue_lengths: Optional[Dict[str, int]] = None,
               next_phase: Optional[int] = None,
               next_time_in_phase: Optional[int] = None,
               departed: Optional[int] = None,
               total_vehicles_served: Optional[Dict[str, int]] = None):
        """
        Update agent with enhanced reward and state representation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Raw reward
            next_state: Next state
            queue_lengths: Current queue lengths
            phase: Current phase
            time_in_phase: Time in phase
            done: Whether episode is finished
            next_queue_lengths: Next queue lengths (after step)
            next_phase: Next phase (after step)
            next_time_in_phase: Next time in phase (after step)
            departed: Number of departed vehicles in this step
            total_vehicles_served: Total vehicles served per direction
        """
        # Determine actual departed count
        dep_count = departed if departed is not None else (1 if reward > 0 else 0)
        
        # Calculate enhanced reward
        enhanced_reward = self.reward_calculator.calculate_reward(
            queue_lengths, departed=dep_count, switch_action=action,
            phase=phase, total_vehicles_served=total_vehicles_served
        )
        
        # Update state representation
        if self.state_rep:
            self.state_rep.update_history(queue_lengths, action, phase)
        
        # Update RL agent
        if self.agent_type == 'neural':
            if self._last_enhanced_state is not None:
                enhanced_state = self._last_enhanced_state
            else:
                enhanced_state = self.state_rep.get_enhanced_state(
                    queue_lengths, phase, time_in_phase
                )
            
            # Compute actual next state features from next-state parameters
            n_queues = next_queue_lengths if next_queue_lengths is not None else queue_lengths
            n_phase = next_phase if next_phase is not None else phase
            n_time = next_time_in_phase if next_time_in_phase is not None else (time_in_phase + 1)
            
            next_enhanced_state = self.state_rep.get_enhanced_state(
                n_queues, n_phase, n_time
            )
            
            self.rl_agent.remember(enhanced_state, action, enhanced_reward,
                                 next_enhanced_state, done)
            self.rl_agent.replay()
        else:
            self.rl_agent.update(state, action, enhanced_reward, next_state, done)
        
        # Track enhanced rewards
        self.enhanced_rewards_history.append(enhanced_reward)
    
    def update_learning_parameters(self, episode: int, total_episodes: int):
        """Update learning parameters using adaptive scheduler."""
        recent_reward = np.mean(self.enhanced_rewards_history[-10:]) if self.enhanced_rewards_history else 0
        self.learning_scheduler.update_parameters(episode, total_episodes, recent_reward)
        
        # Update agent parameters
        if self.agent_type == 'neural':
            alpha, epsilon = self.learning_scheduler.get_parameters()
            self.rl_agent.epsilon = epsilon
        else:
            alpha, epsilon = self.learning_scheduler.get_parameters()
            self.rl_agent.learning_rate = alpha
            self.rl_agent.set_epsilon(epsilon)
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.rl_agent.reset_episode()
        if self.state_rep:
            self.state_rep.reset_episode()
        self.reward_calculator.reset_episode()
        self._last_enhanced_state = None
    
    def get_statistics(self) -> Dict:
        """Alias for get_comprehensive_statistics."""
        return self.get_comprehensive_statistics()
        
    def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics from all components."""
        stats = {
            'agent_type': self.agent_type,
            'rl_agent_stats': self.rl_agent.get_statistics(),
            'learning_scheduler_stats': {
                'current_alpha': self.learning_scheduler.current_alpha,
                'current_epsilon': self.learning_scheduler.current_epsilon
            },
            'episode_count': len(self.episode_rewards),
            'avg_enhanced_reward': np.mean(self.enhanced_rewards_history) if self.enhanced_rewards_history else 0
        }
        
        return stats
    
    def save(self, filepath: str):
        """Save enhanced agent state."""
        if self.agent_type == 'neural':
            self.rl_agent.save(filepath)
        else:
            self.rl_agent.save(filepath)
        
        # Save enhanced components
        enhanced_data = {
            'agent_type': self.agent_type,
            'reward_calculator_config': self.reward_calculator.config,
            'learning_scheduler_state': {
                'current_alpha': self.learning_scheduler.current_alpha,
                'current_epsilon': self.learning_scheduler.current_epsilon,
                'reward_history': self.learning_scheduler.reward_history
            },
            'episode_rewards': self.episode_rewards[-1000:],
            'enhanced_rewards_history': self.enhanced_rewards_history[-1000:]
        }
        
        enhanced_path = filepath.replace('.pkl', '_enhanced.pkl')
        with open(enhanced_path, 'wb') as f:
            pickle.dump(enhanced_data, f)
            
    def load(self, filepath: str):
        """Load agent state from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent file not found: {filepath}")
        
        # Detect if model is neural or tabular
        is_neural = False
        if TORCH_AVAILABLE:
            try:
                data = torch.load(filepath, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'q_network_state' in data:
                    is_neural = True
                    if self.agent_type != 'neural':
                        self.agent_type = 'neural'
                        self.rl_agent = NeuralNetworkAgent(
                            state_size=data.get('state_size', 13),
                            action_size=data.get('action_size', 2)
                        )
                        if self.state_rep is None:
                            self.state_rep = enhanced_rewards.EnhancedStateRepresentation()
                    self.rl_agent.load(filepath)
            except Exception:
                is_neural = False
        
        if not is_neural:
            if self.agent_type != 'tabular':
                self.agent_type = 'tabular'
                self.rl_agent = SimpleQLearningAgent()
                self.state_rep = None
            self.rl_agent.load(filepath)
            
        enhanced_path = filepath.replace('.pkl', '_enhanced.pkl')
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, 'rb') as f:
                    enhanced_data = pickle.load(f)
                    if 'reward_calculator_config' in enhanced_data:
                        self.reward_calculator.config = enhanced_data['reward_calculator_config']
                    if 'learning_scheduler_state' in enhanced_data:
                        sched = enhanced_data['learning_scheduler_state']
                        self.learning_scheduler.current_alpha = sched.get('current_alpha', self.learning_scheduler.current_alpha)
                        self.learning_scheduler.current_epsilon = sched.get('current_epsilon', self.learning_scheduler.current_epsilon)
                        self.learning_scheduler.reward_history = sched.get('reward_history', [])
                    if 'episode_rewards' in enhanced_data:
                        self.episode_rewards = enhanced_data['episode_rewards']
                    if 'enhanced_rewards_history' in enhanced_data:
                        self.enhanced_rewards_history = enhanced_data['enhanced_rewards_history']
            except Exception as e:
                print(f"Note: Could not load enhanced metadata ({e}), base agent loaded.")


if __name__ == "__main__":
    # Test enhanced hybrid agent
    print("Testing enhanced hybrid agent...")
    
    # Test neural network agent
    neural_config = {
        'state_size': 13,
        'learning_rate': 0.001,
        'epsilon_start': 0.5,
        'reward_config': {
            'w_wait': 1.0,
            'w_switch': 0.2,
            'w_throughput': 0.1,
            'w_fairness': 0.3
        }
    }
    
    enhanced_agent = EnhancedHybridAgent('neural', neural_config)
    
    # Test with dummy data
    state = (1, 2, 0, 3, 1, 2)
    queue_lengths = {'N': 5, 'S': 3, 'E': 8, 'W': 2}
    phase = 0
    time_in_phase = 15
    
    action = enhanced_agent.get_action(state, queue_lengths, phase, time_in_phase)
    print(f"Selected action: {action}")
    
    # Update with dummy reward
    enhanced_agent.update(state, action, -10.0, state, queue_lengths, phase, time_in_phase)
    
    # Get statistics
    stats = enhanced_agent.get_comprehensive_statistics()
    print(f"Agent statistics: {stats}")
    
    print("Enhanced hybrid agent test completed!")
