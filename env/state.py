"""
SystemState: the single source of truth for one episode.
Observation construction lives here; mutation lives in transitions.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from env.models import Task, Resource, TaskStatus


@dataclass
class SystemState:
    current_step:  int
    tasks:         dict[str, Task]
    resources:     dict[str, Resource]
    total_steps:   int               = 50
    action_history: list[dict]       = field(default_factory=list)
    event_log:     list[dict]        = field(default_factory=list)

    # ── Derived views ──────────────────────────────────────────────

    @property
    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    @property
    def delayed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.DELAYED]

    @property
    def active_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values()
                if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)]

    @property
    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

    @property
    def failed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

    @property
    def available_resources(self) -> list[Resource]:
        return [r for r in self.resources.values() if r.is_available]

    @property
    def system_load(self) -> float:
        if not self.resources:
            return 0.0
        return sum(r.utilization for r in self.resources.values()) / len(self.resources)

    # ── Observation ────────────────────────────────────────────────

    def to_observation(self) -> dict:
        """The dict an agent receives each step."""
        return {
            "current_step":       self.current_step,
            "total_steps":        self.total_steps,
            "steps_remaining":    self.total_steps - self.current_step,
            "pending_tasks":      [t.to_dict() for t in self.pending_tasks],
            "delayed_tasks":      [t.to_dict() for t in self.delayed_tasks],
            "active_tasks":       [t.to_dict() for t in self.active_tasks],
            "completed_count":    len(self.completed_tasks),
            "completed_task_ids": [t.task_id for t in self.completed_tasks],
            "failed_count":       len(self.failed_tasks),
            "resources":          [r.to_dict() for r in self.resources.values()],
            "system_load":        round(self.system_load, 3),
            "recent_events":      self.event_log[-6:],
        }