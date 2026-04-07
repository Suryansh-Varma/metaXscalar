"""
Structured logging — JSONL for machine consumption, CSV for spreadsheets.
"""

from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results")


def _ensure_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def log_episode(result: dict) -> None:
    """Append one episode result to the JSONL log."""
    _ensure_dir()
    path = RESULTS_DIR / "episodes.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str) + "\n")


def log_summary(summary: dict, agent_name: str) -> None:
    """Append an aggregated summary row to the CSV."""
    _ensure_dir()
    path   = RESULTS_DIR / "summary.csv"
    is_new = not path.exists()
    row    = {"agent": agent_name, "timestamp": datetime.now().isoformat(), **summary}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_json(data: dict, filename: str) -> str:
    _ensure_dir()
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return str(path)