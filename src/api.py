"""
api.py — Stdin/stdout bridge. Entry point for Tauri subprocess.
Tauri spawns: python api.py --task "description"
All communication is JSON lines on stdout.
Commands from Tauri arrive on stdin as JSON lines.
"""

from __future__ import annotations
import sys
import json
import argparse
import logging
import os

# configure logging to stderr only — stdout is reserved for Tauri bridge
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def bridge(type_: str, **data):
    """Send a structured JSON line to Tauri."""
    print(json.dumps({"type": type_, **data}), flush=True)


def cmd_run(args):
    from src.agent.runner import AgentRunner
    bridge("ready", {"version": "0.1.0"})
    runner = AgentRunner(
        model_str=os.environ.get("AGENT_MODEL", "ollama:mistral"),
        data_dir=args.data_dir,
    )
    runner.run(args.task, max_steps=args.max_steps)


def cmd_stats(args):
    from src.agent.memory import Memory
    from src.agent.tasks import TaskManager
    from src.training.pipeline import TrajectoryCapture

    mem = Memory(
        kv_path=f"{args.data_dir}/memory/kv.json",
        vector_path=f"{args.data_dir}/memory/vectors",
    )
    tasks = TaskManager(path=f"{args.data_dir}/tasks.jsonl")
    traj = TrajectoryCapture(events_path=f"{args.data_dir}/trajectories/events.jsonl")

    bridge("stats", {
        "memory": mem.stats(),
        "tasks": tasks.stats(),
        "training": traj.stats(),
    })


def cmd_dataset(args):
    from src.training.pipeline import TrajectoryCapture, DatasetFormatter

    capture = TrajectoryCapture(events_path=f"{args.data_dir}/trajectories/events.jsonl")
    samples = capture.extract_samples()
    formatter = DatasetFormatter(output_dir=f"{args.data_dir}/datasets")

    if args.format == "alpaca":
        path = formatter.to_alpaca(samples)
    elif args.format == "sharegpt":
        path = formatter.to_sharegpt(samples)
    else:
        path = formatter.to_jsonl(samples)

    bridge("dataset", {
        "path": path,
        "summary": formatter.summary(samples),
    })


def main():
    parser = argparse.ArgumentParser(description="Agent Framework Bridge")
    parser.add_argument("--data-dir", default="./data")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run")
    run_p.add_argument("task", help="Task description")
    run_p.add_argument("--max-steps", type=int, default=200)

    sub.add_parser("stats")

    ds_p = sub.add_parser("dataset")
    ds_p.add_argument("--format", choices=["jsonl", "alpaca", "sharegpt"], default="jsonl")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "dataset":
        cmd_dataset(args)
    else:
        bridge("error", {"error": "No command specified. Use: run | stats | dataset"})
        sys.exit(1)


if __name__ == "__main__":
    main()
