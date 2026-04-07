"""
Dense, multi-factor reward engine.
Every component is isolated, named, and independently logged.
Output is always normalised to [0.0, 1.0].
"""

from __future__ import annotations
from dataclasses import dataclass, field
from env.state import SystemState
from env.models import TaskStatus


# Component weights — must sum to 1.0
WEIGHTS = {
    "completion":  0.35,
    "priority":    0.25,
    "efficiency":  0.15,
    "timeliness":  0.15,
    "overload":    0.10,
}

DELAY_PENALTY_PER_STEP  = 0.02
OVERLOAD_PENALTY_PER_UNIT = 0.03
INVALID_ACTION_PENALTY    = 0.05

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


@dataclass
class RewardBreakdown:
    completion_score:  float = 0.0
    priority_score:    float = 0.0
    efficiency_score:  float = 0.0
    timeliness_score:  float = 0.0
    overload_penalty:  float = 0.0
    conflict_penalty:  float = 0.0
    total:             float = 0.0
    details:           dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "completion":  round(self.completion_score, 4),
            "priority":    round(self.priority_score, 4),
            "efficiency":  round(self.efficiency_score, 4),
            "timeliness":  round(self.timeliness_score, 4),
            "overload":    round(self.overload_penalty, 4),
            "conflict":    round(self.conflict_penalty, 4),
            "total":       round(self.total, 4),
            "details":     self.details,
        }


def compute_step_reward(
    state: SystemState,
    action_valid: bool,
) -> RewardBreakdown:
    """Compute reward for one environment step."""
    b = RewardBreakdown()
    total = len(state.tasks)
    if total == 0:
        return b

    completed = state.completed_tasks

    # 1. Completion ratio
    b.completion_score = len(completed) / total

    # 2. Priority-weighted completion
    max_weight = sum(t.priority.value for t in state.tasks.values())
    got_weight = sum(t.priority.value for t in completed)
    b.priority_score = got_weight / max_weight if max_weight else 0.0

    # 3. Efficiency — sweet spot is 60–85% utilisation
    utils = [r.utilization for r in state.resources.values()]
    avg_u = sum(utils) / len(utils) if utils else 0.0
    if avg_u <= 0.85:
        b.efficiency_score = avg_u / 0.85
    else:
        b.efficiency_score = max(0.0, 1.0 - (avg_u - 0.85) * 3)
    b.details["avg_util"] = round(avg_u, 3)

    # 4. Timeliness — graduated penalty per late step
    total_late_steps = 0
    late_count = 0
    for t in state.tasks.values():
        if t.status == TaskStatus.FAILED:
            overshoot = max(0, state.current_step - t.deadline)
            total_late_steps += overshoot
            late_count += 1
        elif t.status == TaskStatus.COMPLETED and t.completed_at and t.completed_at > t.deadline:
            total_late_steps += t.completed_at - t.deadline
            late_count += 1
    timeliness_penalty = min(1.0, total_late_steps * DELAY_PENALTY_PER_STEP)
    b.timeliness_score = 1.0 - timeliness_penalty
    b.details["late_tasks"] = late_count

    # 5. Overload penalty
    excess = sum(
        max(0, r.current_load - r.capacity)
        for r in state.resources.values()
    )
    b.overload_penalty = min(1.0, excess * OVERLOAD_PENALTY_PER_UNIT)
    b.details["overload_units"] = excess

    # 6. Invalid action penalty
    if not action_valid:
        b.conflict_penalty = INVALID_ACTION_PENALTY

    raw = (
        WEIGHTS["completion"]  * b.completion_score
        + WEIGHTS["priority"]  * b.priority_score
        + WEIGHTS["efficiency"]* b.efficiency_score
        + WEIGHTS["timeliness"]* b.timeliness_score
        - WEIGHTS["overload"]  * b.overload_penalty
        - b.conflict_penalty
    )
    b.total = max(0.0, min(1.0, raw))
    return b


def compute_final_reward(state: SystemState) -> RewardBreakdown:
    """Terminal score evaluated over the full episode."""
    return compute_step_reward(state, action_valid=True)