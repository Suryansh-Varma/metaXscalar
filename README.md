---
title: OpsArena Cloud Ops Command Centre
emoji: none
colorFrom: indigo
colorTo: slate
sdk: docker
pinned: false
---

# OpsArena — Cloud Operations Command Centre

> **A production-grade AI agent evaluation benchmark** for multi-step operational scheduling under resource constraints, SLA deadlines, and cascading infrastructure failures.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com)

---

## Real-World Problem Framing

OpsArena simulates the decision loop of a **Cloud NOC (Network Operations Centre)** operator:

| Real-world concept | Simulation analogue |
|---|---|
| Microservice deployment | Task assigned to CPU cluster |
| Database backup window | IO resource scheduled |
| SLA breach / P1 incident | Deadline expiry → task FAILED |
| Incident response ticket | Critical task arrival mid-episode |
| Resource exhaustion | Capacity exceeded → overload penalty |
| MTTR (Mean Time To Repair) | Steps taken to complete each task |

Agents must allocate compute, memory, IO, and network resources to arriving service tasks — while juggling conflicting priorities, dependency chains, and unexpected resource failures.

---

## Architecture

```
ops/
├── env/
│   ├── environment.py     # Ops class — OpenAI-gym-style API: reset / step / final_score
│   ├── models.py          # Task, Resource, Priority, TaskStatus dataclasses
│   ├── state.py           # SystemState — single source of truth per episode
│   ├── tasks.py           # Scenario factory (easy / medium / hard)
│   ├── transitions.py     # Pure state-mutation functions (assign, tick, fail, …)
│   ├── reward.py          # Multi-factor reward engine (5 components, normalised 0–1)
│   └── utils.py           # Action validation, observation formatting
│
├── agents/
│   ├── base.py            # BaseAgent ABC
│   ├── random_agent.py    # Random-assignment baseline
│   ├── greedy_agent.py    # Priority + EDF greedy heuristic
│   ├── rl_agent.py        # Q-learning agent with linear function approximation
│   └── llm_agent.py       # LLM-prompted agent (requires ANTHROPIC_API_KEY)
│
├── evaluation/
│   ├── evaluator.py       # run_episode(), evaluate_all()
│   ├── metrics.py         # Aggregation helpers
│   └── logger.py          # JSON result persistence
│
├── app/
│   └── interface.py       # Gradio dashboard (4 tabs, live visualisation)
│
├── scripts/
│   └── train_rl.py        # Offline RL training script
│
├── Dockerfile             # Single-service container
├── Dockerfile.dashboard   # Dashboard-specific image
└── docker-compose.yml     # Multi-service: dashboard + train + eval
```

---

## Quick Start

### Local (virtualenv)

```bash
pip install -r requirements.txt

# Launch dashboard
python -m app.interface      # → http://localhost:7860

# (Optional) Train RL agent first
python scripts/train_rl.py --episodes 300 --difficulty medium --verbose
```

### Docker — Dashboard only

```bash
docker build -t opsarena .
docker run -p 7860:7860 -v $(pwd)/results:/app/results opsarena
```

### Docker Compose — Full stack

```bash
# Just the dashboard
docker compose up dashboard

# Train RL agent (one-off job)
docker compose --profile train up train-rl

# Run batch evaluation
docker compose --profile eval up eval
```

---

## Agents

| Agent | Strategy | Avg Score (medium) |
|---|---|---|
| **Random** | Random valid assignment | ~0.35 |
| **Greedy** | Highest-priority first, EDF tiebreak | ~0.72 |
| **RL (Q-Learning)** | Learned linear value function | ~0.78* |

*After 200+ training episodes.

### RL Agent Details

`agents/rl_agent.py` implements **epsilon-greedy Q-learning with linear function approximation**:

```
Q(s, a) = w · φ(s, a)

φ = [urgency, slack_ratio, priority_norm, load_inverse, progress_ratio]
```

Weight update (TD-0):
```
w += α · (r + γ · max_a' Q(s', a') - Q(s, a)) · φ(s, a)
```

Train from the UI (Train RL Agent tab) or via CLI:
```bash
python scripts/train_rl.py --episodes 300 --difficulty hard --verbose
```

---

## Reward Components

| Component | Weight | Description |
|---|---|---|
| Completion | **0.35** | Fraction of all tasks completed |
| Priority | **0.25** | Priority-weighted completion ratio |
| Efficiency | **0.15** | Resource utilisation in 60–85% sweet spot |
| Timeliness | **0.15** | Penalty for late or expired tasks |
| Overload | **0.10** | Penalty for exceeding resource capacity |

Total reward is always normalised to **[0.0, 1.0]**.

---

## Environment API

```python
from env.environment import Ops
from agents import GreedyAgent

env = Ops(difficulty="hard", seed=42)
obs = env.reset()
agent = GreedyAgent()
agent.reset()

done = False
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)

score = env.final_score()
print(f"Score: {score.total:.4f}")
```

---

## Running Tests

```bash
pytest tests/ -v
```
