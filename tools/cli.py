#!/usr/bin/env python
"""
cli.py — Command line interface for agent-framework.
Usage:
  python tools/cli.py run "build a REST API"
  python tools/cli.py stats
  python tools/cli.py dataset --format alpaca
  python tools/cli.py memory query "how to implement auth"
"""

import sys
import os

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import typer
from typing import Optional
from enum import Enum

app = typer.Typer(
    name="agent",
    help="Agent Framework CLI",
    add_completion=False,
)


class DatasetFormat(str, Enum):
    jsonl    = "jsonl"
    alpaca   = "alpaca"
    sharegpt = "sharegpt"


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description for the agent"),
    model: str = typer.Option("ollama:mistral", "--model", "-m", help="Model backend (ollama:mistral | hf:mistral-7b | openai:gpt-4o)"),
    max_steps: int = typer.Option(200, "--max-steps", help="Maximum loop steps"),
    data_dir: str = typer.Option("./data", "--data-dir", help="Data directory"),
):
    """Run the agent on a task."""
    os.environ["AGENT_MODEL"] = model

    from src.agent.runner import AgentRunner
    typer.echo(f"[agent] starting: {task}")
    typer.echo(f"[agent] model: {model}  max_steps: {max_steps}")

    runner = AgentRunner(model_str=model, data_dir=data_dir)
    runner.run(task, max_steps=max_steps)


@app.command()
def stats(
    data_dir: str = typer.Option("./data", "--data-dir"),
):
    """Show memory, task, and training stats."""
    import json
    from src.agent.memory import Memory
    from src.agent.tasks import TaskManager
    from src.training.pipeline import TrajectoryCapture

    mem   = Memory(kv_path=f"{data_dir}/memory/kv.json", vector_path=f"{data_dir}/memory/vectors")
    tasks = TaskManager(path=f"{data_dir}/tasks.jsonl")
    traj  = TrajectoryCapture(events_path=f"{data_dir}/trajectories/events.jsonl")

    typer.echo("\n── MEMORY ──────────────────────────────")
    typer.echo(json.dumps(mem.stats(), indent=2))
    typer.echo("\n── TASKS ───────────────────────────────")
    typer.echo(json.dumps(tasks.stats(), indent=2))
    typer.echo("\n── TRAINING ────────────────────────────")
    typer.echo(json.dumps(traj.stats(), indent=2))


@app.command()
def dataset(
    format: DatasetFormat = typer.Option(DatasetFormat.jsonl, "--format", "-f"),
    data_dir: str = typer.Option("./data", "--data-dir"),
):
    """Export training dataset from captured trajectories."""
    from src.training.pipeline import TrajectoryCapture, DatasetFormatter

    typer.echo(f"[dataset] extracting samples from {data_dir}/trajectories/events.jsonl")
    capture   = TrajectoryCapture(events_path=f"{data_dir}/trajectories/events.jsonl")
    samples   = capture.extract_samples()
    formatter = DatasetFormatter(output_dir=f"{data_dir}/datasets")

    if format == DatasetFormat.alpaca:
        path = formatter.to_alpaca(samples)
    elif format == DatasetFormat.sharegpt:
        path = formatter.to_sharegpt(samples)
    else:
        path = formatter.to_jsonl(samples)

    summary = formatter.summary(samples)
    typer.echo(f"[dataset] exported {summary['n_samples']} samples → {path}")
    typer.echo(f"[dataset] estimated tokens: {summary['estimated_tokens']:,}")


memory_app = typer.Typer(help="Memory operations")
app.add_typer(memory_app, name="memory")


@memory_app.command("query")
def memory_query(
    query: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    data_dir: str = typer.Option("./data", "--data-dir"),
):
    """Query the vector memory store."""
    from src.agent.memory import Memory
    mem = Memory(kv_path=f"{data_dir}/memory/kv.json", vector_path=f"{data_dir}/memory/vectors")
    results = mem.retrieve(query, top_k=top_k)
    typer.echo(f"\n── TOP {top_k} RESULTS FOR: '{query}' ──────────")
    for r in results:
        typer.echo(f"  [{r.get('score', 0):.3f}] {r.get('text', '')[:120]}")


@memory_app.command("store")
def memory_store(
    text: str = typer.Argument(...),
    data_dir: str = typer.Option("./data", "--data-dir"),
):
    """Manually store text in vector memory."""
    from src.agent.memory import Memory
    mem = Memory(kv_path=f"{data_dir}/memory/kv.json", vector_path=f"{data_dir}/memory/vectors")
    idx = mem.store(text)
    typer.echo(f"[memory] stored at index {idx}")


if __name__ == "__main__":
    app()
