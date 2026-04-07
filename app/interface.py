"""
OpsArenaEnv — Cloud Operations Command Centre
============================================
A real-time AI evaluation dashboard for multi-agent operational scheduling.
Includes FastAPI endpoints for OpenEnv automated compliance.
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import Optional, Dict, Any

import gradio as gr
from fastapi import FastAPI, Body

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.environment import Ops
from env.utils import format_observation, action_to_str
from agents import GreedyAgent, RandomAgent, RLAgent, BaseAgent
from evaluation.evaluator import evaluate_all

# ── API Setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="OpsArena OpenEnv API")
_GLOBAL_ENV: Optional[Ops] = None

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/reset")
def reset_env(payload: Dict[str, Any] = Body(...)):
    global _GLOBAL_ENV
    difficulty = payload.get("difficulty", "medium")
    seed = payload.get("seed", 42)
    _GLOBAL_ENV = Ops(difficulty=difficulty, seed=seed)
    obs = _GLOBAL_ENV.reset()
    return {"observation": obs, "status": "reset_success"}

@app.post("/step")
def step_env(action: Dict[str, Any] = Body(...)):
    global _GLOBAL_ENV
    if _GLOBAL_ENV is None:
        return {"error": "Environment not initialized. Call /reset first."}, 400
    obs, reward, done, info = _GLOBAL_ENV.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

# ── Gradio Helpers ──────────────────────────────────────────────────────────

_RL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rl_weights.json")

def _get_agent(agent_name: str, seed: int = 42):
    if agent_name == "Greedy": return GreedyAgent()
    if agent_name == "Random": return RandomAgent(seed=seed)
    if agent_name == "RL (Q-Learning)":
        if os.path.exists(_RL_WEIGHTS_PATH):
            agent = RLAgent.load(_RL_WEIGHTS_PATH)
            agent.epsilon = 0.0  # Force exploitation in UI demo
            return agent
        return RLAgent(seed=seed)
    return GreedyAgent()

def _priority_label(p: int) -> str: return {1: "LOW", 2: "MED", 3: "HIGH", 4: "CRIT"}.get(p, str(p))
def _priority_color(p: int) -> str: return {1: "#6b7280", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444"}.get(p, "#9ca3af")

def _build_task_board_html(obs: dict) -> str:
    cols = {
        "pending": ("Pending", obs.get("pending_tasks", []) + obs.get("delayed_tasks", []), "#1e3a5f"),
        "active":  ("In Progress", obs.get("active_tasks", []), "#1a3a2f"),
        "done":    ("Done", [], "#1a2e1a"),
    }
    completed_count = obs.get("completed_count", 0)
    failed_count = obs.get("failed_count", 0)
    html = """<style>
    .task-board {{ display: flex; gap: 14px; font-family: 'Inter', sans-serif; }}
    .kb-col {{ flex: 1; border-radius: 12px; padding: 12px; }}
    .kb-header {{ font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #94a3b8; margin-bottom: 10px; }}
    .task-card {{ border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); font-size: 12.5px; color: #e2e8f0; }}
    .task-name {{ font-weight: 600; margin-bottom: 4px; }}
    .task-meta {{ font-size: 11px; color: #94a3b8; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 1px 6px; font-size: 10.5px; font-weight: 700; margin-right: 4px; }}
    .prog-bar {{ height: 4px; border-radius: 2px; background: rgba(255,255,255,0.1); margin-top: 6px; }}
    .prog-fill {{ height: 100%; border-radius: 2px; background: #22d3ee; }}
    .summary-pills {{ display: flex; gap: 8px; margin-top: 6px; font-family: 'Inter', sans-serif; }}
    .pill {{ border-radius: 6px; padding: 4px 10px; font-size: 11.5px; font-weight: 600; }}
    </style>
    <div class="summary-pills">
      <div class="pill" style="background:#1e3a5f;color:#60a5fa">⏳ {pending} Pending</div>
      <div class="pill" style="background:#1a3a2f;color:#34d399">⚡ {active} Active</div>
      <div class="pill" style="background:#1e2d1e;color:#86efac">✓ {done} Done</div>
      <div class="pill" style="background:#3a1a1a;color:#fca5a5">✗ {fail} Failed</div>
      <div class="pill" style="background:#1e1e2e;color:#94a3b8">Step {s}/{t}</div>
    </div><br><div class="task-board">""".format(
        pending=len(cols["pending"][1]), active=len(cols["active"][1]), 
        done=completed_count, fail=failed_count, 
        s=obs.get("current_step","?"), t=obs.get("total_steps","?")
    )
    for col_key, (title, tasks, bg) in cols.items():
        html += f'<div class="kb-col" style="background:{bg}20;border:1px solid {bg}60"><div class="kb-header">{title}</div>'
        if col_key == "done":
            if completed_count: html += f'<div class="task-card"><div class="task-name" style="color:#34d399">✓ {completed_count} completed</div></div>'
            if failed_count: html += f'<div class="task-card"><div class="task-name" style="color:#f87171">✗ {failed_count} failed</div></div>'
        else:
            for t in tasks:
                color = _priority_color(t.get("priority", 1))
                html += f"""<div class="task-card"><div class="task-name">{t.get("name","?")}</div><div class="task-meta">
                    <span class="badge" style="background:{color}33;color:{color}">{_priority_label(t.get("priority", 1))}</span>
                    deadline: <b>{t.get("deadline","?")}</b> &nbsp; dur: {t.get("duration", 1)}</div>
                    <div class="prog-bar"><div class="prog-fill" style="width:{min(100, int(t.get("progress",0)/max(t.get("duration",1),1)*100))}%"></div></div></div>"""
        html += "</div>"
    html += "</div>"
    return html

def _build_resource_meters_html(obs: dict) -> str:
    resources = obs.get("resources", [])
    html = """<style>.res-grid { display: flex; gap: 10px; flex-wrap: wrap; font-family: 'Inter', sans-serif; }
    .res-card { flex: 1; min-width: 160px; border-radius: 10px; padding: 12px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
    .res-name { font-size: 12px; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
    .res-util { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .res-bar { height: 6px; border-radius: 3px; background: rgba(255,255,255,0.08); margin-bottom: 6px; }
    .res-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
    .res-meta { font-size: 11px; color: #64748b; }</style><div class="res-grid">"""
    for r in resources:
        util = r.get("utilization", 0)
        pct = int(util * 100)
        color = "#22d3ee" if pct < 60 else "#34d399" if pct <= 85 else "#f59e0b" if pct <= 100 else "#ef4444"
        html += f"""<div class="res-card"><div class="res-name">{r.get("name","?")}</div><div class="res-util" style="color:{color}">{pct}%</div>
          <div class="res-bar"><div class="res-fill" style="width:{min(100,pct)}%;background:{color}"></div></div>
          <div class="res-meta">{r.get("current_load","?")}/{r.get("capacity","?")} load</div></div>"""
    return html + "</div>"

# ── Gradio Handlers ────────────────────────────────────────────────────────

def _run_full_episode(difficulty: str, agent_name: str, seed: int):
    agent = _get_agent(agent_name, seed=int(seed))
    env = Ops(difficulty=difficulty, seed=int(seed))
    obs = env.reset()
    rewards = []
    done = False
    while not done:
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        rewards.append(r)
    final = env.final_score().to_dict()
    return _build_task_board_html(obs), _build_resource_meters_html(obs), f"Final Score: {final['total']:.4f}\nDifficulty: {difficulty}"

def _train_rl_agent(episodes: int, difficulty: str):
    if os.path.exists(_RL_WEIGHTS_PATH):
        agent = RLAgent.load(_RL_WEIGHTS_PATH)
        log = f"🔄 Loaded existing weights. Resuming training from episode {agent._episode_count}...\n"
    else:
        agent = RLAgent()
        log = f"🚀 Starting fresh training for {episodes} episodes on '{difficulty}'...\n"
    
    scores = agent.train(n_episodes=int(episodes), difficulty=difficulty)
    agent.save(_RL_WEIGHTS_PATH)
    log += f"✅ Training complete. Final Average Score: {sum(scores[-10:])/10:.4f}\n"
    log += f"Weights saved to {_RL_WEIGHTS_PATH}"
    return log

def _run_batch_eval(agent_name: str):
    agent = _get_agent(agent_name)
    log = f"📊 Evaluating {agent_name} across all difficulties...\n"
    results = evaluate_all(agent)
    log += f"Overall Avg Score: {results['overall_avg']}"
    return results, log

# ── Gradio Layout ──────────────────────────────────────────────────────────

_CSS = "body { background: #020817; color: #e2e8f0; }"
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="OpsArena — Cloud Ops") as demo:
    gr.HTML("<h1 style='color:#818cf8'>⚡ OpsArena — Cloud Ops Command Centre</h1>")
    with gr.Tabs():
        with gr.Tab("🎯 Single Episode"):
            with gr.Row():
                diff = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
                agt = gr.Dropdown(["Greedy", "Random", "RL (Q-Learning)"], value="Greedy", label="Agent")
                sd = gr.Number(value=42, label="Seed", precision=0)
                btn = gr.Button("▶ Run Episode", variant="primary")
            b_board = gr.HTML()
            b_res = gr.HTML()
            b_out = gr.Textbox(label="Logs", lines=2)
            btn.click(_run_full_episode, [diff, agt, sd], [b_board, b_res, b_out])

        with gr.Tab("🧠 Train RL Agent"):
            gr.Markdown("Train the Q-Learning agent offline to optimize resource scheduling.")
            with gr.Row():
                rl_eps = gr.Slider(50, 1000, value=200, step=50, label="Episodes")
                rl_diff = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
                tr_btn = gr.Button("🚀 Start Training", variant="primary")
            tr_log = gr.Textbox(label="Training Output", lines=10)
            tr_btn.click(_train_rl_agent, [rl_eps, rl_diff], [tr_log])

        with gr.Tab("📊 Benchmarking"):
            gr.Markdown("Compare agents across a standard set of seeds (42, 123, 777).")
            be_agt = gr.Dropdown(["Greedy", "Random", "RL (Q-Learning)"], value="Greedy", label="Agent")
            be_btn = gr.Button("📈 Run Benchmark", variant="primary")
            be_log = gr.Textbox(label="Benchmark Results", lines=10)
            be_btn.click(_run_batch_eval, [be_agt], [gr.JSON(), be_log])

# Mount Gradio into FastAPI
demo.theme = theme
demo.css = _CSS
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
