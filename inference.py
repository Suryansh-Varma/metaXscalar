"""
Inference script for automated evaluation.
Strictly follows the [START], [STEP], and [END] format defined in the sample.
"""

import os
import sys
import time
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import Ops
from agents import GreedyAgent, RandomAgent, LLMAgent, RLAgent

# ── Configuration ──────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "OpsArenaEnv"

# ── Logging Helpers ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", 
        flush=True
    )

# ── Agent Factory ──────────────────────────────────────────────────────────

def get_agent(name):
    """Retrieve agent based on name or type."""
    name = name.lower()
    if name == "greedy":
        return GreedyAgent()
    if name == "random":
        return RandomAgent(seed=42)
    if name == "rl":
        weights = "results/rl_weights.json"
        if os.path.exists(weights):
            return RLAgent.load(weights)
        return RLAgent()
    if name == "llm":
        return LLMAgent()
    return GreedyAgent()

# ── Execution Logic ────────────────────────────────────────────────────────

def run_episode(difficulty: str, agent_name: str, seed: int):
    """Run a single episode and emit strictly formatted logs."""
    agent = get_agent(agent_name)
    env = Ops(difficulty=difficulty, seed=seed)
    
    obs = env.reset()
    agent.reset()
    
    task_id = f"{difficulty}_{seed}"
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    done = False
    
    while not done:
        try:
            action_dict = agent.act(obs)
            # action_str for logging
            action_str = f"{action_dict.get('type','noop')}"
            if action_dict.get('task_id'):
                action_str += f"({action_dict['task_id']})"
                
            obs, reward, done, info = env.step(action_dict)
            
            rewards.append(reward)
            steps_taken += 1
            
            log_step(
                step=steps_taken, 
                action=action_str, 
                reward=reward, 
                done=done, 
                error=None
            )
        except Exception as e:
            log_step(
                step=steps_taken + 1, 
                action="error", 
                reward=0.0, 
                done=True, 
                error=str(e)
            )
            done = True

    final = env.final_score()
    # Normalize score to [0, 1] as required. final.total is already normalized.
    score = final.total
    success = score >= 0.7  # Example threshold for success
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

def main():
    agent_name = os.environ.get("AGENT_NAME", "greedy")
    difficulties = ["easy", "medium", "hard"]
    seeds = [42] # Use subset for standard run, or expand as needed
    
    for diff in difficulties:
        for seed in seeds:
            run_episode(diff, agent_name, seed)

if __name__ == "__main__":
    main()
