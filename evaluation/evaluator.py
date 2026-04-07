from __future__ import annotations
import time

from agents.base import BaseAgent
from env.environment import Ops
from evaluation.metrics import episode_summary, aggregate
from evaluation import logger


def run_episode(
    agent: BaseAgent,
    difficulty: str,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    env = Ops(difficulty=difficulty, seed=seed)
    obs = env.reset()
    agent.reset()

    step_rewards: list[float] = []
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward)

        if verbose:
            ar = info["action_result"]
            print(
                f"  step {obs['current_step']:3d}  "
                f"r={reward:.3f}  "
                f"{ar['action_str'] if ar['success'] else 'invalid: ' + ar['reason']}"
            )

    final = env.final_score()
    state = env.state

    result = {
        "difficulty": difficulty,
        "seed": seed,
        "agent": agent.name(),
        "total_steps": len(step_rewards),
        "avg_step_reward": sum(step_rewards) / max(len(step_rewards), 1),
        "final_score": final.total,
        "completed_tasks": len(getattr(state, "completed_tasks", [])),
        "failed_tasks": len(getattr(state, "failed_tasks", [])),
        "total_tasks": len(state.tasks),
        "reward_breakdown": final.to_dict(),
    }

    logger.log_episode(result)
    return result


def evaluate_all(
    agent: BaseAgent,
    difficulties: list[str] | None = None,
    seeds: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    difficulties = difficulties or ["easy", "medium", "hard"]
    seeds = seeds or [42, 123, 777]

    all_results: dict[str, dict] = {}

    print(f"\n{'='*56}")
    print(f"  OpsArenaEnv evaluation  —  agent: {agent.name()}")
    print(f"{'='*56}\n")

    all_scores: list[float] = []

    for diff in difficulties:
        print(f"[{diff.upper()}]")
        episode_results: list[dict] = []

        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)

            t0 = time.perf_counter()
            result = run_episode(agent, diff, seed=seed, verbose=verbose)
            elapsed = time.perf_counter() - t0

            ep = episode_summary(result)

            print(
                f"score={ep['final_score']:.3f}  "
                f"completion={ep['completion_rate']:.0%}  "
                f"({elapsed:.1f}s)"
            )

            episode_results.append(result)

        agg = aggregate(episode_results)

        all_results[diff] = {
            "episodes": episode_results,
            **agg
        }

        all_scores.extend(r["final_score"] for r in episode_results)

        print(
            f"  -> avg={agg['avg_final_score']:.4f} "
            f"± {agg['std_final_score']:.4f}  "
            f"completion={agg['avg_completion_rate']:.0%}\n"
        )

    overall = sum(all_scores) / max(len(all_scores), 1)
    all_results["overall_avg"] = round(overall, 4)

    print(f"{'='*56}")
    print(f"  Overall average: {overall:.4f}")
    print(f"{'='*56}\n")

    logger.log_summary({"overall_avg": round(overall, 4)}, agent.name())

    path = logger.save_json(all_results, f"eval_{agent.name()}.json")
    print(f"Full results -> {path}\n")

    return all_results