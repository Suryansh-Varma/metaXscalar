"""
RL Agent — Tabular Q-learning with linear function approximation.

Uses a feature vector derived from the observation to estimate Q-values
for each candidate (task, resource) pair, then acts epsilon-greedily.

Training loop (offline):
    agent = RLAgent()
    agent.train(n_episodes=500, difficulty="medium")
    agent.save("rl_weights.json")

Inference:
    agent = RLAgent.load("rl_weights.json")
    action = agent.act(observation)
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Optional

from agents.base import BaseAgent


# ── Feature engineering ───────────────────────────────────────────────────

def _task_features(task: dict, obs: dict) -> list[float]:
    """Encode a single (task, env) pair into a fixed-length feature vector."""
    steps_remaining = obs.get("steps_remaining", 1) or 1
    total_steps = obs.get("total_steps", 50) or 50
    deadline = task.get("deadline", total_steps)
    duration = task.get("duration", 1) or 1
    priority = task.get("priority", 1)
    system_load = obs.get("system_load", 0.0)
    completed = obs.get("completed_count", 0)
    total_tasks = (
        len(obs.get("pending_tasks", []))
        + len(obs.get("delayed_tasks", []))
        + len(obs.get("active_tasks", []))
        + completed
    ) or 1

    urgency = 1.0 - min(1.0, (deadline - obs.get("current_step", 0)) / max(steps_remaining, 1))
    slack_ratio = max(0.0, (deadline - obs.get("current_step", 0) - duration)) / total_steps
    priority_norm = priority / 4.0
    load_inv = 1.0 - system_load
    progress_ratio = completed / total_tasks

    return [urgency, slack_ratio, priority_norm, load_inv, progress_ratio]


FEATURE_DIM = 5  # must match length returned by _task_features


# ── Q-learning with linear approximation ─────────────────────────────────

class RLAgent(BaseAgent):
    """
    Epsilon-greedy Q-learning agent with linear function approximation.

    Q(s, a) = w · φ(s, a)   where φ is a hand-crafted feature vector.

    Learns to schedule tasks by mapping urgency, priority, slack, and
    resource load into an action-value estimate, updated via TD(0).
    """

    def __init__(
        self,
        alpha: float = 0.05,      # learning rate
        gamma: float = 0.95,      # discount factor
        epsilon: float = 0.25,    # initial exploration
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)
        self._weights = [0.0] * FEATURE_DIM
        self._episode_count = 0
        self._last_obs: Optional[dict] = None
        self._last_features: Optional[list[float]] = None
        self._total_reward: float = 0.0
        self._trained = False

    # ── Inference ─────────────────────────────────────────────────────────

    def act(self, observation: dict) -> dict:
        candidates = self._build_candidates(observation)
        if not candidates:
            return {"type": "noop"}

        if self._trained and self._rng.random() > self.epsilon:
            # Exploit: pick highest Q-value candidate
            best = max(candidates, key=lambda c: self._q(c["features"]))
        else:
            # Explore: random candidate
            best = self._rng.choice(candidates)

        self._last_obs = observation
        self._last_features = best["features"]

        return {
            "type": "assign",
            "task_id": best["task_id"],
            "resource_id": best["resource_id"],
        }

    def update(self, reward: float, next_obs: dict, done: bool) -> None:
        """TD(0) weight update. Call after every env.step() during training."""
        if self._last_features is None:
            return

        # Bootstrap next Q-value
        next_candidates = self._build_candidates(next_obs)
        if next_candidates and not done:
            next_q = max(self._q(c["features"]) for c in next_candidates)
        else:
            next_q = 0.0

        current_q = self._q(self._last_features)
        td_error = reward + self.gamma * next_q - current_q

        # Gradient of linear Q: w += alpha * td_error * phi
        for i in range(FEATURE_DIM):
            self._weights[i] += self.alpha * td_error * self._last_features[i]

        self._total_reward += reward

        if done:
            self._episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self._trained = True

    def reset(self) -> None:
        self._last_obs = None
        self._last_features = None
        self._total_reward = 0.0

    def name(self) -> str:
        eps = f"ε={self.epsilon:.3f}"
        trained = f"ep={self._episode_count}" if self._trained else "untrained"
        return f"RLAgent({trained},{eps})"

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        n_episodes: int = 300,
        difficulty: str = "medium",
        verbose: bool = False,
    ) -> list[float]:
        """
        Run full training loop within the agent.
        Returns list of final scores per episode.
        """
        from env.environment import Ops

        scores: list[float] = []
        seeds = list(range(n_episodes))

        for ep_i, seed in enumerate(seeds):
            env = Ops(difficulty=difficulty, seed=seed)
            obs = env.reset()
            self.reset()
            done = False

            while not done:
                action = self.act(obs)
                next_obs, reward, done, _ = env.step(action)
                self.update(reward, next_obs, done)
                obs = next_obs

            final = env.final_score()
            scores.append(final.total)

            if verbose and (ep_i + 1) % 50 == 0:
                avg = sum(scores[-50:]) / 50
                print(
                    f"  Episode {ep_i+1:4d}/{n_episodes}  "
                    f"avg_score={avg:.4f}  ε={self.epsilon:.3f}"
                )

        self._trained = True
        return scores

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "weights": self._weights,
            "epsilon": self.epsilon,
            "episode_count": self._episode_count,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RLAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(
            alpha=data.get("alpha", 0.05),
            gamma=data.get("gamma", 0.95),
        )
        agent._weights = data["weights"]
        agent.epsilon = data.get("epsilon", 0.02)
        agent._episode_count = data.get("episode_count", 0)
        agent._trained = True
        return agent

    # ── Private helpers ───────────────────────────────────────────────────

    def _q(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self._weights, features))

    def _build_candidates(self, obs: dict) -> list[dict]:
        """Enumerate all valid (task, resource) assignments from observation."""
        candidates = []
        pending = obs.get("pending_tasks", []) + obs.get("delayed_tasks", [])
        free_resources = [r for r in obs.get("resources", []) if r.get("available")]

        for task in pending:
            # Skip tasks with unmet dependencies
            if task.get("dependencies"):
                continue
            for res in free_resources:
                if str(res.get("type", "")) == str(task.get("required_resource", "")):
                    feats = _task_features(task, obs)
                    candidates.append({
                        "task_id": task["task_id"],
                        "resource_id": res["resource_id"],
                        "features": feats,
                    })
        return candidates
