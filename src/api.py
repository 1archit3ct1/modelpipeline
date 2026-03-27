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
from pathlib import Path

# Ensure the root workspace is loaded in the path so deep imports like 'src.agent' always resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# configure logging to stderr only — stdout is reserved for Tauri bridge
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def bridge(type_: str, data: dict = None, **kwargs):
    """Send a structured JSON line to Tauri."""
    combined = data or {}
    combined.update(kwargs)
    print(json.dumps({"type": type_, **combined}), flush=True)

class BridgeLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if record.name != __name__:
                bridge("backend_log", {"text": msg})
        except Exception:
            pass

bridge_handler = BridgeLogHandler()
bridge_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logging.getLogger().addHandler(bridge_handler)

def health_check(data_dir):
    """Test all component imports and emit events so GUI flips them GREEN."""
    checks = {
        "art-runner":       "src.agent.runner",
        "art-actions":      "src.agent.actions",
        "art-executor":     "src.environment.executor",
        "art-memory":       "src.agent.memory",
        "art-state":        "src.agent.state",
        "art-graph":        "src.agent.graph",
        "art-hooks":        "src.agent.hooks",
        "art-trajectory":   "src.training.pipeline",
        "art-dataset":      "src.training.pipeline",
        "art-tasks":        "src.agent.tasks",
        "art-model-router": "src.agent.model",
        "art-bridge":       "src.api",
    }
    for art_id, module_path in checks.items():
        try:
            __import__(module_path)
            bridge("event", {"event": "component.verified", "payload": {"component": art_id}})
        except Exception as e:
            bridge("event", {"event": "component.failed", "payload": {"component": art_id, "error": str(e)}})

    # Check embedder separately (heavier import)
    try:
        import src.vectorstore.embedder
        bridge("event", {"event": "component.verified", "payload": {"component": "art-embedder"}})
    except Exception as e:
        bridge("event", {"event": "component.failed", "payload": {"component": "art-embedder", "error": str(e)}})
        logger.error(f"Embedder failed to load: {e}")

    # Check vector store
    try:
        from src.vectorstore.store import VectorStore
        bridge("event", {"event": "component.verified", "payload": {"component": "art-vectorstore"}})
    except Exception as e:
        bridge("event", {"event": "component.failed", "payload": {"component": "art-vectorstore", "error": str(e)}})
        logger.error(f"VectorStore failed to load: {e}")

    # Report existing training data
    events_path = f"{data_dir}/trajectories/events.jsonl"
    sample_count = 0
    if os.path.exists(events_path):
        with open(events_path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "training.sample":
                        sample_count += 1
                except Exception:
                    pass
    if sample_count > 0:
        bridge("stats", {"training": {"training_samples": sample_count}})


def cmd_run(args):
    from src.agent.runner import AgentRunner
    bridge("ready", {"version": "0.1.0"})
    health_check(args.data_dir)
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
    formatter = DatasetFormatter(output_dir=f"{args.data_dir}/datasets")

    if args.format == "dpo":
        dpo_pairs = capture.extract_dpo_pairs()
        path = formatter.to_dpo(dpo_pairs)
        bridge("dataset", {
            "path": path,
            "summary": {"n_dpo_pairs": len(dpo_pairs)},
        })
    elif args.format == "episodic":
        episodes = capture.extract_episodes()
        path = formatter.to_episodic(episodes)
        bridge("dataset", {
            "path": path,
            "summary": {"n_episodes": len(episodes)},
        })
    else:
        # jsonl, alpaca, sharegpt
        samples = capture.extract_samples(min_quality_score=args.min_quality)
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
    ds_p.add_argument("--format", choices=["jsonl", "alpaca", "sharegpt", "dpo", "episodic"], default="jsonl")
    ds_p.add_argument("--min-quality", type=float, default=0.0, help="Minimum quality score (0.0-1.0)")

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
