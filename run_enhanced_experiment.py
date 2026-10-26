#!/usr/bin/env python3
"""Run TrafficAI training and evaluation experiment pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_command(cmd: list[str], description: str) -> bool:
    """Run a subprocess command and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Success")
        if result.stdout:
            print("Output tail:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed with exit code {e.returncode}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False


def main() -> None:
    """Parse experiment options, train model, and run paired evaluation."""
    parser = argparse.ArgumentParser(description="Run TrafficAI experiment pipeline")
    parser.add_argument("--agent_type", type=str, default="tabular", choices=["tabular", "neural"],
                        help="RL agent type")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--episode_length", type=int, default=900, help="Episode length in seconds")
    parser.add_argument("--quick_test", action="store_true", help="Run with reduced episodes for quick verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--policy_output", default=os.path.join("models", "tabular_policy.json"),
                        help="Path for trained policy export")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.episodes = min(args.episodes, 10)
        args.episode_length = min(args.episode_length, 300)
        print("Quick test mode: reduced parameters for fast testing")
    
    print("TrafficAI Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Settings: episodes={args.episodes}, episode_length={args.episode_length}, seed={args.seed}")
    
    # Step 1: Training
    train_cmd = [
        sys.executable, "enhanced_train.py",
        "--agent_type", args.agent_type,
        "--episodes", str(args.episodes),
        "--episode_length", str(args.episode_length),
        "--seed", str(args.seed),
        "--policy_output", args.policy_output,
        "--scenarios", "balanced", "ns_skew", "heavy", "spike"
    ]
    
    if not run_command(train_cmd, "Training"):
        print("Training failed.")
        sys.exit(1)
    
    # Step 2: Evaluation
    if os.path.exists(args.policy_output):
        eval_cmd = [
            sys.executable, "evaluation.py",
            "--agent", args.policy_output,
            "--episode_length", str(args.episode_length),
            "--runs", "5" if args.quick_test else "20",
            "--scenarios", "balanced", "heavy", "ns_heavy", "spike"
        ]
        run_command(eval_cmd, "Paired Evaluation")
    else:
        print(f"Policy file not found at {args.policy_output}")
    
    print(f"\n{'='*70}")
    print("Experiment completed.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
