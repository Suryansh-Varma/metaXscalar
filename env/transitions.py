"""
Pure state-mutation functions.
Each transition takes (state, ...) and returns a (new_state, event_str) tuple.
No side effects — all randomness goes through an explicit rng argument.
"""

from __future__ import annotations
import copy
import random
from env.state import SystemState
from env.models import Task, Resource, TaskStatus, Priority


def apply_assign(state: SystemState, task_id: str, resource_id: str, step: int) -> tuple[SystemState, str]:
    task     = state.tasks[task_id]
    resource = state.resources[resource_id]

    task.status              = TaskStatus.IN_PROGRESS
    task.assigned_resource_id = resource_id
    task.started_at          = step
    resource.assigned_task_id = task_id
    resource.current_load   += task.priority.value  # higher priority = heavier load unit

    event = f"Step {step}: [{task.priority.name}] {task.name} -> {resource.name}"
    return state, event


def apply_delay(state: SystemState, task_id: str, extra_steps: int, step: int) -> tuple[SystemState, str]:
    task = state.tasks[task_id]
    task.deadline += extra_steps
    task.status    = TaskStatus.DELAYED
    event = f"Step {step}: DELAY {task.name} +{extra_steps} steps (new deadline={task.deadline})"
    return state, event


def apply_reprioritize(state: SystemState, task_id: str, new_priority: int, step: int) -> tuple[SystemState, str]:
    task     = state.tasks[task_id]
    old      = task.priority.name
    task.priority = Priority(new_priority)
    event = f"Step {step}: REPRIORITIZE {task.name} {old} -> {task.priority.name}"
    return state, event


def tick_tasks(state: SystemState, step: int) -> list[str]:
    """Advance all in-progress tasks by one step; complete them when done."""
    events: list[str] = []
    for task in list(state.tasks.values()):
        if task.status != TaskStatus.IN_PROGRESS:
            continue
        task.progress += 1
        if task.progress >= task.duration:
            task.status       = TaskStatus.COMPLETED
            task.completed_at = step
            res = state.resources.get(task.assigned_resource_id or "")
            if res:
                res.assigned_task_id = None
                res.current_load     = max(0, res.current_load - task.priority.value)
            events.append(f"Step {step}: DONE {task.name}")
    return events


def expire_deadlines(state: SystemState, step: int) -> list[str]:
    """Fail any non-terminal task whose deadline has passed."""
    events: list[str] = []
    for task in state.tasks.values():
        if task.is_terminal:
            continue
        if step > task.deadline:
            task.status = TaskStatus.FAILED
            res = state.resources.get(task.assigned_resource_id or "")
            if res and res.assigned_task_id == task.task_id:
                res.assigned_task_id = None
                res.current_load     = max(0, res.current_load - task.priority.value)
            events.append(f"Step {step}: EXPIRED {task.name} (deadline={task.deadline})")
    return events


def inject_arrivals(state: SystemState, arrivals: list[tuple[int, Task]], step: int) -> list[str]:
    """Add dynamically-arriving tasks at the correct step."""
    events: list[str] = []
    for arrival_step, task in arrivals:
        if arrival_step == step and task.task_id not in state.tasks:
            state.tasks[task.task_id] = copy.deepcopy(task)
            events.append(f"Step {step}: ARRIVAL [{task.priority.name}] {task.name} deadline={task.deadline}")
    return events


def simulate_failures(state: SystemState, rng: random.Random, step: int) -> list[str]:
    """Randomly fail resources; reset their assigned task to PENDING."""
    events: list[str] = []
    for res in state.resources.values():
        if res.failure_rate <= 0 or not res.assigned_task_id:
            continue
        if rng.random() < res.failure_rate:
            task = state.tasks.get(res.assigned_task_id)
            if task:
                task.status              = TaskStatus.PENDING
                task.assigned_resource_id = None
                task.progress            = max(0, task.progress - 1)
            res.assigned_task_id = None
            res.current_load     = max(0, res.current_load - 1)
            events.append(f"Step {step}: FAILURE {res.name} — task reset")
    return events
