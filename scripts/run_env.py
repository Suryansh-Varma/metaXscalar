"""
CLI for manual episode play. Useful for debugging and sanity checks.

Usage:
    python scripts/run_env.py --difficulty hard
"""

import argparse
import json
from env.environment import OpsArenaEnvironment
from env.utils import format_observation


def main():
    parser = argparse.ArgumentParser(description="Play an OpsArena episode manually.")
    parser.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    env = OpsArenaEnvironment(difficulty=args.difficulty, seed=args.seed)
    obs = env.reset()
    done = False

    print(f"\nOpsArenaEnv  |  difficulty={args.difficulty}  seed={args.seed}")
    print("Enter a JSON action each step. Type 'quit' to exit.\n")

    while not done:
        print(format_observation(obs))
        raw = input("\nAction> ").strip()
        if raw.lower() in ("quit", "q", "exit"):
            break
        try:
            action = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON error: {e}")
            continue

        obs, reward, done, info = env.step(action)
        ar = info["action_result"]
        print(f"\n  reward={reward:.4f}  valid={ar['success']}")
        if not ar["success"]:
            print(f"  reason: {ar['reason']}")
        print()

    final = env.final_score()
    print(f"\n--- Episode over ---")
    print(f"Final score : {final.total:.4f}")
    print(f"Completed   : {len(env.state.completed_tasks)} / {len(env.state.tasks)}")


if __name__ == "__main__":
    main()