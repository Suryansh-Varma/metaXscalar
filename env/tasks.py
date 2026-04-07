"""
Scenario factory — three difficulty tiers.
Each scenario returns a ready-to-use dict consumed by OpsArenaEnvironment.
"""

from __future__ import annotations
import copy
from env.models import Task, Resource, Priority, ResourceType, TaskStatus


# ── Helpers ────────────────────────────────────────────────────────────────

def _task(task_id, name, priority, rtype, duration, deadline, created_at=0, deps=None) -> Task:
    return Task(
        task_id=task_id, name=name, priority=priority,
        required_resource=rtype, duration=duration,
        deadline=deadline, created_at=created_at,
        dependencies=deps or [],
    )


def _resource(resource_id, name, rtype, capacity, failure_rate=0.0) -> Resource:
    return Resource(
        resource_id=resource_id, name=name,
        resource_type=rtype, capacity=capacity,
        failure_rate=failure_rate,
    )


# ── Easy ───────────────────────────────────────────────────────────────────

def get_easy_scenario() -> dict:
    """
    3 tasks, 3 dedicated resources, no conflicts, no dependencies.
    A greedy assign-immediately strategy scores ~0.90+.
    """
    tasks = {
        "t1": _task("t1", "Auth service deploy",  Priority.HIGH,   ResourceType.CPU,    4, 20),
        "t2": _task("t2", "Database backup",      Priority.MEDIUM, ResourceType.IO,     6, 25),
        "t3": _task("t3", "Cache warmup",         Priority.LOW,    ResourceType.MEMORY, 3, 30),
    }
    resources = {
        "r1": _resource("r1", "CPU node A",   ResourceType.CPU,    2),
        "r2": _resource("r2", "Disk array",   ResourceType.IO,     2),
        "r3": _resource("r3", "Memory pool",  ResourceType.MEMORY, 2),
    }
    return {"tasks": tasks, "resources": resources, "total_steps": 30, "difficulty": "easy",
            "dynamic_arrivals": []}


# ── Medium ─────────────────────────────────────────────────────────────────

def get_medium_scenario() -> dict:
    """
    6 tasks competing for 3 resources; dependency chain on IO path.
    Greedy fails — priority-aware scheduling with dependency resolution required.
    """
    tasks = {
        "t1": _task("t1", "API gateway deploy",    Priority.CRITICAL, ResourceType.CPU,    5, 15),
        "t2": _task("t2", "ML inference batch",    Priority.HIGH,     ResourceType.CPU,    8, 20),
        "t3": _task("t3", "Log aggregation",       Priority.LOW,      ResourceType.IO,     4, 35),
        "t4": _task("t4", "DB schema migration",   Priority.HIGH,     ResourceType.IO,     6, 22, deps=["t3"]),
        "t5": _task("t5", "Frontend asset build",  Priority.MEDIUM,   ResourceType.CPU,    3, 25),
        "t6": _task("t6", "Metrics roll-up",       Priority.LOW,      ResourceType.MEMORY, 5, 40),
    }
    resources = {
        "r1": _resource("r1", "CPU cluster",   ResourceType.CPU,    2),
        "r2": _resource("r2", "Disk array",    ResourceType.IO,     1),
        "r3": _resource("r3", "Memory pool",   ResourceType.MEMORY, 2),
    }
    return {"tasks": tasks, "resources": resources, "total_steps": 45, "difficulty": "medium",
            "dynamic_arrivals": []}


# ── Hard ───────────────────────────────────────────────────────────────────

# Tasks that inject mid-episode; (arrival_step, Task)
_HARD_ARRIVALS: list[tuple[int, Task]] = [
    (5,  _task("t4", "Incident response",   Priority.CRITICAL, ResourceType.CPU,     5, 14, created_at=5)),
    (8,  _task("t5", "Analytics pipeline",  Priority.MEDIUM,   ResourceType.IO,      7, 25, created_at=8)),
    (10, _task("t6", "Audit log export",    Priority.HIGH,     ResourceType.IO,      5, 20, created_at=10, deps=["t2"])),
    (12, _task("t7", "Capacity rebalance",  Priority.HIGH,     ResourceType.CPU,     4, 22, created_at=12)),
    (15, _task("t8", "Emergency failover",  Priority.CRITICAL, ResourceType.NETWORK, 3, 20, created_at=15)),
]


def get_hard_scenario() -> dict:
    """
    3 initial tasks + 5 dynamic arrivals; resource failures; cascading deadlines.
    Requires lookahead, adaptive reprioritisation, and failure recovery.
    Naive strategies collapse.
    """
    initial_tasks = {
        "t1": _task("t1", "Security patch deploy", Priority.CRITICAL, ResourceType.CPU,    6, 12),
        "t2": _task("t2", "Checkpoint backup",     Priority.MEDIUM,   ResourceType.IO,     8, 30),
        "t3": _task("t3", "Session cache init",    Priority.LOW,      ResourceType.MEMORY, 4, 40),
    }
    resources = {
        "r1": _resource("r1", "CPU cluster A",  ResourceType.CPU,     2, failure_rate=0.05),
        "r2": _resource("r2", "Disk array",     ResourceType.IO,      1, failure_rate=0.03),
        "r3": _resource("r3", "Memory pool",    ResourceType.MEMORY,  2, failure_rate=0.02),
        "r4": _resource("r4", "Network bus",    ResourceType.NETWORK, 1, failure_rate=0.04),
    }
    return {
        "tasks":            initial_tasks,
        "resources":        resources,
        "total_steps":      50,
        "difficulty":       "hard",
        "dynamic_arrivals": [(s, copy.deepcopy(t)) for s, t in _HARD_ARRIVALS],
    }


SCENARIOS: dict[str, callable] = {
    "easy":   get_easy_scenario,
    "medium": get_medium_scenario,
    "hard":   get_hard_scenario,
}


def load_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS)}")
    return SCENARIOS[name]()