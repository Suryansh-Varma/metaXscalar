"""
Unit tests for OpsArenaEnv.
Run with: pytest tests/test_env.py -v
"""

import pytest
from env.environment import Ops
from env.models import TaskStatus, Priority, ResourceType
from env.utils import validate_action
from env.state import SystemState
from env.tasks import load_scenario


# ── Scenario loading ────────────────────────────────────────────────────────

@pytest.mark.parametrize("diff", ["easy", "medium", "hard"])
def test_load_scenario(diff):
    s = load_scenario(diff)
    assert "tasks" in s and "resources" in s
    assert s["total_steps"] > 0
    assert len(s["tasks"]) > 0


# ── Reset ───────────────────────────────────────────────────────────────────

def test_reset_returns_observation():
    env = Ops("easy", seed=42)
    obs = env.reset()
    assert "current_step" in obs
    assert obs["current_step"] == 0
    assert "pending_tasks" in obs
    assert "resources" in obs


def test_reset_is_reproducible():
    env1 = Ops("medium", seed=42)
    env2 = Ops("medium", seed=42)
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert obs1["pending_tasks"] == obs2["pending_tasks"]


# ── Step ────────────────────────────────────────────────────────────────────

def test_noop_advances_step():
    env = Ops("easy", seed=42)
    env.reset()
    obs, reward, done, info = env.step({"type": "noop"})
    assert obs["current_step"] == 1
    assert 0.0 <= reward <= 1.0


def test_invalid_action_penalised():
    env = Ops("easy", seed=42)
    env.reset()
    _, reward_valid,   _, _    = env.step({"type": "noop"})
    env2 = Ops("easy", seed=42)
    env2.reset()
    _, reward_invalid, _, info = env2.step({"type": "assign", "task_id": "NONE", "resource_id": "NONE"})
    assert not info["action_result"]["success"]
    assert reward_invalid <= reward_valid


def test_valid_assign():
    env = Ops("easy", seed=42)
    obs = env.reset()
    pending   = obs["pending_tasks"]
    resources = obs["resources"]
    assert pending and resources

    task = pending[0]
    res  = next((r for r in resources if r["type"] == task["required_resource"] and r["available"]), None)
    if res is None:
        pytest.skip("No compatible free resource in easy scenario")

    _, reward, _, info = env.step({
        "type":        "assign",
        "task_id":     task["task_id"],
        "resource_id": res["resource_id"],
    })
    assert info["action_result"]["success"]
    assert reward > 0.0


def test_episode_terminates():
    env = Ops("easy", seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step({"type": "noop"})
        steps += 1
        assert steps <= 100, "Episode did not terminate"


# ── Validate action ─────────────────────────────────────────────────────────

def test_validate_assign_type_mismatch():
    scenario = load_scenario("easy")
    state    = SystemState(current_step=0, **{k: v for k, v in scenario.items()
                                               if k in ("tasks","resources","total_steps")})
    task_id  = list(state.tasks.keys())[0]
    task     = state.tasks[task_id]

    # Find a resource with a DIFFERENT type
    wrong_res = next(
        (r for r in state.resources.values() if r.resource_type != task.required_resource),
        None,
    )
    if wrong_res is None:
        pytest.skip("All resources have the same type in easy scenario")

    valid, reason = validate_action(
        {"type": "assign", "task_id": task_id, "resource_id": wrong_res.resource_id},
        state,
    )
    assert not valid
    assert "mismatch" in reason.lower()


def test_validate_unknown_action_type():
    scenario = load_scenario("easy")
    state    = SystemState(current_step=0, **{k: v for k, v in scenario.items()
                                               if k in ("tasks","resources","total_steps")})
    valid, reason = validate_action({"type": "teleport"}, state)
    assert not valid


# ── Reward ───────────────────────────────────────────────────────────────────

def test_final_score_in_range():
    env = Ops("medium", seed=42)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step({"type": "noop"})
    score = env.final_score()
    assert 0.0 <= score.total <= 1.0


def test_completion_improves_score():
    """Completing tasks must improve the final score vs all-noop baseline."""
    # All noop
    env_base = Ops("easy", seed=42)
    env_base.reset()
    done = False
    while not done:
        _, _, done, _ = env_base.step({"type": "noop"})
    base_score = env_base.final_score().total

    # Greedy assign
    from agents.greedy_agent import GreedyAgent
    env_act = Ops("easy", seed=42)
    obs = env_act.reset()
    agent = GreedyAgent()
    done  = False
    while not done:
        action = agent.act(obs)
        obs, _, done, _ = env_act.step(action)
    active_score = env_act.final_score().total

    assert active_score >= base_score