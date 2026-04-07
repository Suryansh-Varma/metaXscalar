"""
Greedy agent — always assigns the highest-priority actionable task
to the first compatible free resource. Breaks on dependency conflicts and
resource contention; serves as the non-LLM performance ceiling.
"""

from __future__ import annotations
from agents.base import BaseAgent


class GreedyAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        pending   = observation.get("pending_tasks", []) + observation.get("delayed_tasks", [])
        resources = observation.get("resources", [])
        completed = {
            t["task_id"] for t in observation.get("active_tasks", [])
            # We only have completed_count, not IDs, so we track through history elsewhere.
            # Greedy skips tasks with pending dependencies.
        }

        # Sort by priority descending, then by deadline ascending (EDF tiebreak)
        actionable = sorted(
            pending,
            key=lambda t: (-t["priority"], t["deadline"]),
        )

        free_resources = {r["resource_id"]: r for r in resources if r["available"]}

        for task in actionable:
            # Skip tasks with unmet dependencies (we can't check completion from obs alone,
            # so we skip any task that declares dependencies — conservative but safe)
            if task.get("dependencies"):
                continue
            for rid, res in free_resources.items():
                if res["type"] == task["required_resource"]:
                    return {
                        "type":        "assign",
                        "task_id":     task["task_id"],
                        "resource_id": rid,
                    }

        return {"type": "noop"}