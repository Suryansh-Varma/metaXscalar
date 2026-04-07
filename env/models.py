"""
Core domain models: Task, Resource, Priority, enums.
Pure data — no business logic lives here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING     = "pending"
    ASSIGNED    = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    DELAYED     = "delayed"


class Priority(int, Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class ResourceType(str, Enum):
    CPU     = "cpu"
    MEMORY  = "memory"
    IO      = "io"
    NETWORK = "network"


@dataclass
class Task:
    task_id:             str
    name:                str
    priority:            Priority
    required_resource:   ResourceType
    duration:            int           # steps needed to complete
    deadline:            int           # absolute step deadline
    created_at:          int           # step when task first appeared
    status:              TaskStatus    = TaskStatus.PENDING
    assigned_resource_id: Optional[str] = None
    started_at:          Optional[int] = None
    completed_at:        Optional[int] = None
    progress:            int           = 0
    dependencies:        list[str]     = field(default_factory=list)

    @property
    def slack(self) -> int:
        """Steps remaining before the deadline becomes unbeatable."""
        return self.deadline - self.created_at - self.duration

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "task_id":           self.task_id,
            "name":              self.name,
            "priority":          self.priority.value,
            "required_resource": self.required_resource.value,
            "duration":          self.duration,
            "deadline":          self.deadline,
            "status":            self.status.value,
            "progress":          self.progress,
            "dependencies":      self.dependencies,
        }


@dataclass
class Resource:
    resource_id:   str
    name:          str
    resource_type: ResourceType
    capacity:      int
    current_load:  int           = 0
    assigned_task_id: Optional[str] = None
    failure_rate:  float         = 0.0   # per-step failure probability (hard mode)

    @property
    def is_available(self) -> bool:
        return self.assigned_task_id is None and self.current_load < self.capacity

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.current_load / self.capacity

    @property
    def is_overloaded(self) -> bool:
        return self.current_load > self.capacity

    def to_dict(self) -> dict:
        return {
            "resource_id":   self.resource_id,
            "name":          self.name,
            "type":          self.resource_type.value,
            "capacity":      self.capacity,
            "current_load":  self.current_load,
            "utilization":   round(self.utilization, 3),
            "available":     self.is_available,
            "assigned_task": self.assigned_task_id,
        }