from __future__ import annotations
import statistics
from typing import List, Dict


# ----------------------------
# Single Episode Summary
# ----------------------------
def episode_summary(result: Dict) -> Dict:
    """
    Summarise a single episode result.

    Expected keys in `result`:
    - difficulty
    - seed
    - final_score
    - completed_tasks
    - failed_tasks
    - total_tasks
    - avg_step_reward
    - total_steps (optional)
    - avg_delay (optional)
    - resource_utilization (optional)
    - invalid_actions (optional)
    """

    total_tasks = max(result.get("total_tasks", 1), 1)
    total_steps = max(result.get("total_steps", 1), 1)

    return {
        # Identity
        "difficulty": result.get("difficulty"),
        "seed": result.get("seed"),

        # Core performance
        "final_score": round(result.get("final_score", 0.0), 4),
        "completion_rate": round(result.get("completed_tasks", 0) / total_tasks, 4),
        "failure_rate": round(result.get("failed_tasks", 0) / total_tasks, 4),

        # Efficiency
        "avg_step_reward": round(result.get("avg_step_reward", 0.0), 4),
        "reward_per_step": round(result.get("final_score", 0.0) / total_steps, 4),

        # Advanced metrics (safe defaults)
        "avg_delay": round(result.get("avg_delay", 0.0), 4),
        "utilization": round(result.get("resource_utilization", 0.0), 4),
        "invalid_action_rate": round(result.get("invalid_actions", 0) / total_steps, 4),
    }


# ----------------------------
# Aggregate Metrics
# ----------------------------
def aggregate(results: List[Dict]) -> Dict:
    """
    Aggregate multiple episode results.

    Returns:
    - mean, std, min, max for scores
    - avg completion
    - avg delay, utilization
    """

    if not results:
        return {}

    scores = [r.get("final_score", 0.0) for r in results]
    completion_rates = [
        r.get("completed_tasks", 0) / max(r.get("total_tasks", 1), 1)
        for r in results
    ]

    delays = [r.get("avg_delay", 0.0) for r in results]
    utilizations = [r.get("resource_utilization", 0.0) for r in results]

    return {
        "n": len(results),

        # Score stats
        "avg_final_score": round(statistics.mean(scores), 4),
        "std_final_score": round(
            statistics.stdev(scores) if len(scores) > 1 else 0.0, 4
        ),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),

        # Completion
        "avg_completion_rate": round(statistics.mean(completion_rates), 4),

        # Advanced (safe)
        "avg_delay": round(statistics.mean(delays), 4),
        "avg_utilization": round(statistics.mean(utilizations), 4),
    }


# ----------------------------
# Pretty Print (Optional)
# ----------------------------
def print_summary(summary: Dict):
    """Nicely format a single episode summary."""
    print("\n--- Episode Summary ---")
    print(f"Difficulty:        {summary.get('difficulty')}")
    print(f"Seed:              {summary.get('seed')}")
    print(f"Final Score:       {summary.get('final_score')}")
    print(f"Completion Rate:   {summary.get('completion_rate') * 100:.1f}%")
    print(f"Failure Rate:      {summary.get('failure_rate') * 100:.1f}%")
    print(f"Avg Step Reward:   {summary.get('avg_step_reward')}")
    print(f"Reward/Step:       {summary.get('reward_per_step')}")
    print(f"Avg Delay:         {summary.get('avg_delay')}")
    print(f"Utilization:       {summary.get('utilization')}")
    print(f"Invalid Actions:   {summary.get('invalid_action_rate')}")


def print_aggregate(agg: Dict):
    """Nicely format aggregate results."""
    print("\n=== Aggregate Results ===")
    print(f"Episodes:           {agg.get('n')}")
    print(f"Avg Score:          {agg.get('avg_final_score')}")
    print(f"Std Score:          {agg.get('std_final_score')}")
    print(f"Min Score:          {agg.get('min_score')}")
    print(f"Max Score:          {agg.get('max_score')}")
    print(f"Avg Completion:     {agg.get('avg_completion_rate') * 100:.1f}%")
    print(f"Avg Delay:          {agg.get('avg_delay')}")
    print(f"Avg Utilization:    {agg.get('avg_utilization')}")