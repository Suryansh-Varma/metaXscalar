"""
Stateless helpers: formatting, action validation, pretty-printing.
"""

from __future__ import annotations
import json
from env.state import SystemState
from env.models import TaskStatus, Priority


def validate_action(action: dict, state: SystemState) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Covers type checks, resource compatibility, dependency ordering.
    """
    atype = action.get("type")
    valid_types = {"assign", "delay", "reprioritize", "noop"}

    if atype not in valid_types:
        return False, f"Unknown type '{atype}'. Valid: {valid_types}"

    if atype == "assign":
        tid = action.get("task_id", "")
        rid = action.get("resource_id", "")
        if tid not in state.tasks:
            return False, f"Task '{tid}' not found"
        if rid not in state.resources:
            return False, f"Resource '{rid}' not found"

        task = state.tasks[tid]
        res  = state.resources[rid]

        if task.status not in (TaskStatus.PENDING, TaskStatus.DELAYED):
            return False, f"Task '{tid}' has status={task.status.value} — cannot assign"
        if not res.is_available:
            return False, f"Resource '{rid}' is not available (load={res.current_load}/{res.capacity})"
        if task.required_resource != res.resource_type:
            return False, (
                f"Type mismatch: task needs {task.required_resource.value}, "
                f"resource is {res.resource_type.value}"
            )
        for dep_id in task.dependencies:
            dep = state.tasks.get(dep_id)
            if dep and dep.status != TaskStatus.COMPLETED:
                return False, f"Dependency '{dep_id}' not yet completed (status={dep.status.value})"

    elif atype == "reprioritize":
        tid = action.get("task_id", "")
        np  = action.get("new_priority")
        if tid not in state.tasks:
            return False, f"Task '{tid}' not found"
        valid_priorities = {p.value for p in Priority}
        if np not in valid_priorities:
            return False, f"Invalid priority {np}. Use {sorted(valid_priorities)}"

    elif atype == "delay":
        tid = action.get("task_id", "")
        if tid not in state.tasks:
            return False, f"Task '{tid}' not found"
        steps = action.get("steps", 1)
        if not isinstance(steps, int) or steps < 1:
            return False, f"'steps' must be a positive integer, got {steps!r}"

    return True, "ok"


def format_observation(obs: dict) -> str:
    """Human-readable state summary for the UI and agent prompts."""
    lines = [
        f"Step {obs['current_step']} / {obs['total_steps']}  "
        f"({obs['steps_remaining']} remaining)  "
        f"load={obs['system_load']:.0%}",
        "",
    ]

    if obs["pending_tasks"] or obs["delayed_tasks"]:
        lines.append("Actionable tasks:")
        for t in obs["pending_tasks"] + obs["delayed_tasks"]:
            deps = f"  [needs: {', '.join(t['dependencies'])}]" if t["dependencies"] else ""
            lines.append(
                f"  [{t['priority']}* {t['status']}] {t['task_id']} — {t['name']}"
                f"  dur={t['duration']} deadline={t['deadline']}{deps}"
            )
        lines.append("")

    if obs["active_tasks"]:
        lines.append("In progress:")
        for t in obs["active_tasks"]:
            lines.append(
                f"  {t['task_id']} — {t['name']}  {t['progress']}/{t['duration']} steps"
            )
        lines.append("")

    lines.append("Resources:")
    for r in obs["resources"]:
        filled = int(r["utilization"] * 10)
        bar    = "#" * filled + "-" * (10 - filled)
        avail  = "free" if r["available"] else "busy"
        lines.append(
            f"  {r['resource_id']}  {r['name']:16s}  [{bar}] "
            f"{r['current_load']}/{r['capacity']}  {avail}"
        )

    if obs.get("recent_events"):
        lines += ["", "Recent events:"]
        for ev in obs["recent_events"]:
            if isinstance(ev, dict):
                msg = ev.get("data", {}).get("message", "") or ev.get("type", "event")
                lines.append(f"  [{ev.get('step', '?')}] {msg}")
            else:
                lines.append(f"  {ev}")

    return "\n".join(lines)


def action_to_str(action: dict) -> str:
    t = action.get("type", "?")
    if t == "assign":
        return f"assign {action.get('task_id')} -> {action.get('resource_id')}"
    if t == "delay":
        return f"delay {action.get('task_id')} +{action.get('steps', 1)}"
    if t == "reprioritize":
        return f"reprioritize {action.get('task_id')} -> p{action.get('new_priority')}"
    return "noop"


def safe_json(obj) -> str:
    return json.dumps(obj, default=str, indent=2)