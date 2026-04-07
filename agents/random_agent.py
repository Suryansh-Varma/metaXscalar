from __future__ import annotations
import random
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, observation: dict) -> dict:
        pending = observation.get("pending_tasks", [])
        resources = observation.get("resources", [])

        # Filter tasks with no unmet dependencies
        eligible = [
            t for t in pending
            if not t.get("dependencies") or len(t.get("dependencies", [])) == 0
        ]

        # Available resources
        free_res = [r for r in resources if r.get("available")]

        # Try random assignment
        if eligible and free_res:
            task = self._rng.choice(eligible)

            compat = [
                r for r in free_res
                if str(r["type"]) == str(task["required_resource"])
            ]

            if compat:
                return {
                    "type": "assign",
                    "task_id": task["task_id"],
                    "resource_id": self._rng.choice(compat)["resource_id"],
                }

        # fallback
        return {"type": "noop"}