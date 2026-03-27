"""
hooks.py — Event system. Every hook call = one training data point.
The loop emits events. Hooks capture them. Training pipeline consumes the log.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_hooks: Dict[str, List[Callable]] = {}
_event_log_path: Optional[Path] = None


def init(log_dir: str = "./data/trajectories"):
    global _event_log_path
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _event_log_path = Path(log_dir) / "events.jsonl"
    logger.info(f"Hooks initialized. Event log: {_event_log_path}")


def on(event: str, fn: Callable):
    """Register a callback for an event."""
    _hooks.setdefault(event, []).append(fn)


def emit(event: str, payload: Dict[str, Any]):
    """
    Emit an event. Calls all registered callbacks.
    Always appends to event log — this IS the training data.
    """
    record = {
        "event": event,
        "payload": payload,
        "timestamp": time.time(),
    }

    # write to training log
    if _event_log_path is not None:
        with open(_event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # call registered handlers
    for fn in _hooks.get(event, []):
        try:
            fn(payload)
        except Exception as e:
            logger.error(f"Hook error on event '{event}': {e}")

    # also emit to stdout bridge for Tauri
    _emit_to_bridge(record)


def _emit_to_bridge(record: Dict):
    """Print JSON line to stdout so Tauri frontend can read it."""
    try:
        print(json.dumps({"type": "event", **record}), flush=True)
    except Exception:
        pass


# ── Standard event names ─────────────────────────────────────────────────────

class Event:
    LOOP_START        = "loop.start"
    LOOP_STOP         = "loop.stop"
    LOOP_PAUSE        = "loop.pause"
    LOOP_RESUME       = "loop.resume"
    LOOP_STEP         = "loop.step"

    TASK_CREATED      = "task.created"
    TASK_STARTED      = "task.started"
    TASK_COMPLETED    = "task.completed"
    TASK_FAILED       = "task.failed"

    ACTION_PROPOSED   = "action.proposed"
    ACTION_VALIDATED  = "action.validated"
    ACTION_EXECUTED   = "action.executed"
    ACTION_FAILED     = "action.failed"

    MEMORY_STORED     = "memory.stored"
    MEMORY_RETRIEVED  = "memory.retrieved"
    MEMORY_INDEXED    = "memory.indexed"

    MODEL_CALL        = "model.call"
    MODEL_RESPONSE    = "model.response"
    MODEL_ERROR       = "model.error"

    STATE_SAVED       = "state.saved"
    STATE_LOADED      = "state.loaded"

    GRAPH_UPDATED     = "graph.updated"
    TRAINING_SAMPLE   = "training.sample"
