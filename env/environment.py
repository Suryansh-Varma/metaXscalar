from __future__ import annotations
import random
from typing import Optional, Tuple, Dict, Any

from env.state import SystemState
from env.tasks import load_scenario
from env.reward import compute_step_reward, compute_final_reward, RewardBreakdown
from env.utils import validate_action, action_to_str
from env.models import TaskStatus   # ✅ FIXED: moved import here
from env import transitions as T


class Ops:
    """
    Production-style simulation environment.

    S(t+1) = f(S(t), A(t))
    """

    def __init__(self, difficulty: str = "medium", seed: int = 42):
        self.difficulty = difficulty
        self.seed = seed
        self._rng = random.Random(seed)

        self.state: Optional[SystemState] = None
        self._dynamic_arrivals: list = []

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        self._rng = random.Random(self.seed)
        scenario = load_scenario(self.difficulty)

        self.state = SystemState(
            current_step=0,
            tasks=scenario["tasks"],         # dict[str, Task]
            resources=scenario["resources"], # dict[str, Resource]
            total_steps=scenario["total_steps"],
        )

        self._dynamic_arrivals = scenario.get("dynamic_arrivals", [])

        self._log_event("episode_start", {
            "difficulty": self.difficulty,
            "seed": self.seed
        })

        return self.state.to_observation()

    def get_state(self) -> dict:
        if self.state is None:
            raise RuntimeError("Call reset() before get_state().")
        return self.state.to_observation()

    def step(self, action: dict) -> Tuple[dict, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        step = self.state.current_step

        # 1. Environment dynamics
        self._apply_environment_events(step)

        # 2. Validate + execute action
        is_valid, reason = validate_action(action, self.state)

        if is_valid:
            self.state, event = self._dispatch(action, step)
            self._log_event("action_executed", {
                "action": action,
                "message": event
            })
        else:
            self._log_event("invalid_action", {
                "action": action,
                "reason": reason
            })

        action_result = {
            "step": step,
            "type": action.get("type", "unknown"),
            "success": is_valid,
            "action_str": action_to_str(action),
            "reason": "" if is_valid else reason,
        }

        # 3. Advance time
        self._advance_time(step)

        # 4. Compute reward
        breakdown = compute_step_reward(self.state, is_valid)

        self.state.action_history.append({
            "step": step,
            "action": action,
            "result": action_result,
            "reward": round(breakdown.total, 4),
        })

        # 5. Update step
        self.state.current_step += 1

        # 6. Termination
        done = self._check_done()

        return (
            self.state.to_observation(),
            breakdown.total,
            done,
            {
                "action_result": action_result,
                "reward_breakdown": breakdown.to_dict(),
            },
        )

    def final_score(self) -> RewardBreakdown:
        if self.state is None:
            raise RuntimeError("No episode to score.")
        return compute_final_reward(self.state)

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _apply_environment_events(self, step: int):
        """Inject arrivals + simulate failures."""
        events = T.inject_arrivals(self.state, self._dynamic_arrivals, step)
        self._log_events("task_arrival", events)

        events = T.simulate_failures(self.state, self._rng, step)
        self._log_events("resource_failure", events)

    def _advance_time(self, step: int):
        """Progress tasks and deadlines."""
        events = T.tick_tasks(self.state, step)
        self._log_events("task_progress", events)

        events = T.expire_deadlines(self.state, step)
        self._log_events("deadline", events)

    def _dispatch(self, action: dict, step: int):
        """Route action to transition logic."""
        t = action.get("type")

        if t == "assign":
            return T.apply_assign(
                self.state,
                action["task_id"],
                action["resource_id"],
                step
            )

        if t == "delay":
            return T.apply_delay(
                self.state,
                action["task_id"],
                action.get("steps", 1),
                step
            )

        if t == "reprioritize":
            return T.apply_reprioritize(
                self.state,
                action["task_id"],
                action["new_priority"],
                step
            )

        return self.state, f"Step {step}: noop"

    def _check_done(self) -> bool:
        """Check termination conditions."""

        # 1. Max steps reached
        if self.state.current_step >= self.state.total_steps:
            return True

        # 2. All tasks completed
        if all(
            t.status == TaskStatus.COMPLETED
            for t in self.state.tasks.values()
        ):
            return True

        return False

    def _log_event(self, event_type: str, payload: dict):
        """Structured logging."""
        self.state.event_log.append({
            "step": self.state.current_step,
            "type": event_type,
            "data": payload
        })

    def _log_events(self, event_type: str, messages):
        """Batch log events."""
        for msg in messages:
            self._log_event(event_type, {"message": msg})