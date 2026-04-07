"""
Offline RL training script.

Usage (from project root):
    python scripts/train_rl.py [--episodes 300] [--difficulty medium] [--verbose]

Trains the RLAgent and saves weights to results/rl_weights.json.
"""

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.rl_agent import RLAgent


def main():
    parser = argparse.ArgumentParser(description="Train RLAgent offline.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="results/rl_weights.json", help="Output weights path")
    args = parser.parse_args()

    print(f"\n{'='*56}")
    print(f"  RLAgent Training — {args.difficulty} x {args.episodes} episodes")
    print(f"{'='*56}\n")

    if os.path.exists(args.out):
        print(f"[*] Resuming from existing weights in {args.out}...")
        agent = RLAgent.load(args.out)
    else:
        print(f"[*] Starting fresh training session...")
        agent = RLAgent(epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.02, alpha=0.08)

    scores = agent.train(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        verbose=args.verbose,
    )

    avg_last = sum(scores[-50:]) / min(50, len(scores))
    print(f"\nTraining complete:")
    print(f"  Episodes:     {len(scores)}")
    print(f"  Final avg score (last 50): {avg_last:.4f}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Weights: {agent._weights}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.save(args.out)
    print(f"\nWeights saved to: {args.out}\n")


if __name__ == "__main__":
    main()
