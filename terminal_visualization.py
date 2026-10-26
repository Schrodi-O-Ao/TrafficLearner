import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import time
from datetime import datetime
import utilities as utils


class TerminalVisualizer:
    """Terminal-based visualization and logging for traffic simulation."""
    
    def __init__(self, output_dir: str = "output", save_plots: bool = True):
        """
        Args:
            output_dir: Directory to save plots and logs
            save_plots: Whether to save plots to files
        """
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.plots_dir = os.path.join(output_dir, "plots")
        
        if save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
        
        # Setup matplotlib for non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for Windows compatibility
        plt.style.use('default')
        
        # Initialize plot data storage
        self.episode_data = {
            'rewards': [],
            'wait_times': [],
            'queue_lengths': [],
            'throughputs': [],
            'switches': []
        }
        
        # Terminal display settings
        self.last_terminal_update = 0
        self.terminal_update_interval = 5  # Update terminal every 5 seconds
        
        # Colors for terminal output
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
    
    def update_terminal(self, time_step: int, phase: int, queue_lengths: dict,
                       departed: int, avg_wait: float, switches: int):
        """
        Update terminal display with current status.
        
        Args:
            time_step: Current simulation time step
            phase: Current traffic light phase
            queue_lengths: Dictionary of queue lengths per direction
            departed: Number of vehicles departed in this step
            avg_wait: Average waiting time
            switches: Number of switches so far
        """
        current_time = time.time()
        if current_time - self.last_terminal_update < self.terminal_update_interval:
            return
        
        self.last_terminal_update = current_time
        
        # Clear screen (works on most terminals)
        print('\033[2J\033[H', end='')
        
        # Header
        print(f"{self.colors['bold']}{self.colors['cyan']}Traffic AI Simulation{self.colors['reset']}")
        print("=" * 60)
        
        # Status line
        phase_str = "NS-Green" if phase == 0 else "EW-Green" if phase == 1 else "Yellow" if phase == 2 else "All-Red"
        time_formatted = utils.format_time(time_step)
        
        print(f"{self.colors['bold']}Status:{self.colors['reset']} "
              f"t={time_formatted} | Phase={phase_str} | "
              f"Departed={departed} | Avg Wait={avg_wait:.1f}s | Switches={switches}")
        print()
        
        # Queue visualization
        print(f"{self.colors['bold']}Queue Lengths:{self.colors['reset']}")
        for direction in ['N', 'S', 'E', 'W']:
            queue_len = queue_lengths.get(direction, 0)
            bar = utils.create_queue_bar(queue_len, max_length=20)
            color = self.colors['green'] if queue_len < 5 else self.colors['yellow'] if queue_len < 10 else self.colors['red']
            print(f"  {direction}: {color}{bar}{self.colors['reset']} ({queue_len})")
        
        print()
        
        # Performance metrics
        if len(self.episode_data['wait_times']) > 0:
            recent_wait = np.mean(self.episode_data['wait_times'][-10:])
            recent_queue = np.mean(self.episode_data['queue_lengths'][-10:])
            recent_throughput = np.mean(self.episode_data['throughputs'][-10:])
            
            print(f"{self.colors['bold']}Recent Performance (last 10 episodes):{self.colors['reset']}")
            print(f"  Avg Wait: {recent_wait:.2f}s")
            print(f"  Avg Queue: {recent_queue:.1f}")
            print(f"  Avg Throughput: {recent_throughput:.1f} vehicles/episode")
        
        print()
        print(f"{self.colors['cyan']}Last updated: {datetime.now().strftime('%H:%M:%S')}{self.colors['reset']}")
    
    def update_plots(self, episode_rewards: list, wait_times: list, 
                    queue_lengths: list, throughputs: list, switches: list):
        """
        Update and save training progress plots.
        
        Args:
            episode_rewards: List of episode rewards
            wait_times: List of average wait times per episode
            queue_lengths: List of average queue lengths per episode
            throughputs: List of throughputs per episode
            switches: List of switch counts per episode
        """
        if not self.save_plots:
            return
        
        # Store data
        self.episode_data['rewards'] = episode_rewards
        self.episode_data['wait_times'] = wait_times
        self.episode_data['queue_lengths'] = queue_lengths
        self.episode_data['throughputs'] = throughputs
        self.episode_data['switches'] = switches
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Traffic AI Training Progress', fontsize=16)
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # Episode rewards
        if episode_rewards:
            axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.7)
            axes[0, 0].plot(episodes, utils.moving_average(episode_rewards, 10), 'r-', linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend(['Raw', 'Moving Avg'], loc='best')
        
        # Wait times
        if wait_times:
            axes[0, 1].plot(episodes, wait_times, 'g-', alpha=0.7)
            axes[0, 1].plot(episodes, utils.moving_average(wait_times, 10), 'r-', linewidth=2)
            axes[0, 1].set_title('Average Waiting Time')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Wait Time (s)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(['Raw', 'Moving Avg'], loc='best')
        
        # Queue lengths
        if queue_lengths:
            axes[0, 2].plot(episodes, queue_lengths, 'm-', alpha=0.7)
            axes[0, 2].plot(episodes, utils.moving_average(queue_lengths, 10), 'r-', linewidth=2)
            axes[0, 2].set_title('Average Queue Length')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Queue Length')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend(['Raw', 'Moving Avg'], loc='best')
        
        # Throughput
        if throughputs:
            axes[1, 0].plot(episodes, throughputs, 'c-', alpha=0.7)
            axes[1, 0].plot(episodes, utils.moving_average(throughputs, 10), 'r-', linewidth=2)
            axes[1, 0].set_title('Throughput')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Vehicles per Episode')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(['Raw', 'Moving Avg'], loc='best')
        
        # Switches
        if switches:
            axes[1, 1].bar(episodes, switches, alpha=0.7, color='orange')
            axes[1, 1].set_title('Phase Switches per Episode')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Switches')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Combined performance metric
        if wait_times and queue_lengths:
            # Normalize metrics for comparison
            norm_wait = np.array(wait_times) / max(wait_times) if max(wait_times) > 0 else np.array(wait_times)
            norm_queue = np.array(queue_lengths) / max(queue_lengths) if max(queue_lengths) > 0 else np.array(queue_lengths)
            combined_score = 1 - (norm_wait + norm_queue) / 2  # Higher is better
            
            axes[1, 2].plot(episodes, combined_score, 'purple', alpha=0.7)
            axes[1, 2].plot(episodes, utils.moving_average(combined_score, 10), 'r-', linewidth=2)
            axes[1, 2].set_title('Combined Performance Score')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Score (higher is better)')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend(['Raw', 'Moving Avg'], loc='best')
            axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'enhanced_training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
    
    def create_queue_heatmap(self, queue_history: list, save_path: str = None):
        """
        Create heatmap of queue lengths over time.
        
        Args:
            queue_history: List of queue length dictionaries over time
            save_path: Path to save the heatmap
        """
        if not queue_history:
            return
        
        # Extract queue data
        directions = ['N', 'S', 'E', 'W']
        queue_matrix = []
        
        for queues in queue_history:
            row = [queues.get(dir, 0) for dir in directions]
            queue_matrix.append(row)
        
        queue_matrix = np.array(queue_matrix).T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(queue_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(0, len(queue_history), max(1, len(queue_history)//10)))
        ax.set_xticklabels([f"{i}" for i in range(0, len(queue_history), max(1, len(queue_history)//10))])
        ax.set_yticks(np.arange(len(directions)))
        ax.set_yticklabels(directions)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Direction')
        ax.set_title('Queue Length Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Queue Length')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Queue heatmap saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_wait_time_cdf(self, wait_times: list, save_path: str = None):
        """
        Create CDF plot of waiting times.
        
        Args:
            wait_times: List of waiting times
            save_path: Path to save the CDF plot
        """
        if not wait_times:
            return
        
        # Sort wait times for CDF
        sorted_waits = np.sort(wait_times)
        cdf = np.arange(1, len(sorted_waits) + 1) / len(sorted_waits)
        
        # Create CDF plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_waits, cdf, 'b-', linewidth=2)
        
        ax.set_xlabel('Waiting Time (seconds)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Waiting Time CDF')
        ax.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            idx = int(p * len(sorted_waits) / 100)
            if idx < len(sorted_waits):
                ax.axvline(sorted_waits[idx], color='red', linestyle='--', alpha=0.7)
                ax.text(sorted_waits[idx], 0.1, f'{p}%', rotation=90, va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Wait time CDF saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_episode_summary(self, episode_num: int, scenario: str, 
                            avg_wait: float, avg_queue: float, 
                            throughput: int, switches: int):
        """
        Print episode summary to terminal.
        
        Args:
            episode_num: Episode number
            scenario: Traffic scenario
            avg_wait: Average waiting time
            avg_queue: Average queue length
            throughput: Total throughput
            switches: Number of switches
        """
        print(f"\n{self.colors['bold']}Episode {episode_num} Summary{self.colors['reset']}")
        print(f"Scenario: {scenario}")
        print(f"Average Wait Time: {self.colors['yellow']}{avg_wait:.2f}s{self.colors['reset']}")
        print(f"Average Queue Length: {self.colors['cyan']}{avg_queue:.1f}{self.colors['reset']}")
        print(f"Throughput: {self.colors['green']}{throughput}{self.colors['reset']} vehicles")
        print(f"Phase Switches: {self.colors['magenta']}{switches}{self.colors['reset']}")
        print("-" * 40)
    
    def log_to_file(self, message: str, log_file: str = "simulation.log"):
        """
        Log message to file.
        
        Args:
            message: Message to log
            log_file: Log file name
        """
        log_path = os.path.join(self.output_dir, log_file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    # Test visualizer
    print("Testing terminal visualizer...")
    
    visualizer = TerminalVisualizer(output_dir="output/test_viz")
    
    # Test terminal display
    test_queues = {'N': 3, 'S': 7, 'E': 1, 'W': 12}
    visualizer.update_terminal(
        time_step=450,
        phase=0,
        queue_lengths=test_queues,
        departed=2,
        avg_wait=15.3,
        switches=5
    )
    
    # Test plots with dummy data
    episodes = 50
    dummy_rewards = np.random.randn(episodes).cumsum()
    dummy_waits = 20 + 10 * np.exp(-np.arange(episodes) / 20) + np.random.randn(episodes) * 2
    dummy_queues = 15 + 8 * np.exp(-np.arange(episodes) / 25) + np.random.randn(episodes) * 1.5
    dummy_throughput = 100 + 50 * (1 - np.exp(-np.arange(episodes) / 15)) + np.random.randn(episodes) * 5
    dummy_switches = np.random.poisson(10, episodes)
    
    visualizer.update_plots(
        episode_rewards=dummy_rewards.tolist(),
        wait_times=dummy_waits.tolist(),
        queue_lengths=dummy_queues.tolist(),
        throughputs=dummy_throughput.tolist(),
        switches=dummy_switches.tolist()
    )
    
    # Test episode summary
    visualizer.print_episode_summary(
        episode_num=25,
        scenario="balanced",
        avg_wait=12.5,
        avg_queue=4.2,
        throughput=145,
        switches=8
    )
    
    print("Visualizer test completed!")
