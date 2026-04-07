"""
Greedy agent — always assigns the highest-priority actionable task
to the first compatible free resource. Uses dependency-awareness via
completed_task_ids exposed in the observation.
"""

from __future__ import annotations
from agents.base import BaseAgent


class GreedyAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        pending      = observation.get("pending_tasks", []) + observation.get("delayed_tasks", [])
        resources    = observation.get("resources", [])
        completed_ids = set(observation.get("completed_task_ids", []))

        # Sort by priority descending, then by deadline ascending (EDF tiebreak)
        actionable = sorted(
            pending,
            key=lambda t: (-t["priority"], t["deadline"]),
        )

        free_resources = {r["resource_id"]: r for r in resources if r["available"]}

        for task in actionable:
            # Skip tasks whose dependencies haven't finished yet
            deps = task.get("dependencies", [])
            if deps and not all(d in completed_ids for d in deps):
                continue
            for rid, res in free_resources.items():
                if res["type"] == task["required_resource"]:
                    return {
                        "type":        "assign",
                        "task_id":     task["task_id"],
                        "resource_id": rid,
                    }

        return {"type": "noop"}