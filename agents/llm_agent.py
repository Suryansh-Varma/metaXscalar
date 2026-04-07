"""
LLM Agent — Using OpenAI client with API_BASE_URL and MODEL_NAME.
Compliant with automated evaluation requirements.
"""

from __future__ import annotations
import json
import os
import sys
from openai import OpenAI
from agents.base import BaseAgent
from env.utils import format_observation

SYSTEM_PROMPT = """You are an expert operations engineer scheduling tasks on a resource-constrained system.

Each step you receive the current system state. Respond with exactly ONE JSON action.

Valid actions:
  {"type": "assign",       "task_id": "<id>", "resource_id": "<id>"}
  {"type": "reprioritize", "task_id": "<id>", "new_priority": <1|2|3|4>}
  {"type": "delay",        "task_id": "<id>", "steps": <positive_int>}
  {"type": "noop"}

Rules:
- Only assign tasks with status "pending" or "delayed".
- Resource type must match task's required_resource field exactly.
- Resource must be available (free=true).
- Never assign a task whose dependencies are not yet completed.
- Priority: 1=LOW 2=MEDIUM 3=HIGH 4=CRITICAL. Always serve CRITICAL tasks first.
- Prefer earliest-deadline tasks when priority is equal.

Output ONLY the JSON object. No prose, no markdown, no explanation."""


class LLMAgent(BaseAgent):
    def __init__(self, verbose: bool = False):
        self.api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "missing")
        self.base_url = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
        self.model = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.verbose = verbose

    def act(self, observation: dict) -> dict:
        obs_text = format_observation(observation)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text}
                ],
                max_tokens=256,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            
            # Clean possible markdown
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            
            action = json.loads(raw)
        except Exception as exc:
            if self.verbose:
                print(f"[LLMAgent] Error: {exc}", file=sys.stderr)
            action = {"type": "noop"}
            
        return action