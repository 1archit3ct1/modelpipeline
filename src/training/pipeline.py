"""
training/pipeline.py — Trajectory capture and dataset formatter.
Reads events.jsonl → formats into training pairs → outputs JSONL dataset.
This IS the data flywheel. Every run improves the next model.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrajectoryCapture:
    """
    Reads the live event log and assembles (context, action, result) triplets.
    These become supervised fine-tuning samples.
    """

    def __init__(self, events_path: str = "./data/trajectories/events.jsonl"):
        self.events_path = Path(events_path)

    def load_events(self) -> List[Dict]:
        if not self.events_path.exists():
            return []
        with open(self.events_path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def extract_samples(self) -> List[Dict[str, Any]]:
        """
        Extract training samples from event log.
        Each sample: {prompt, completion, metadata}
        """
        events = self.load_events()
        samples = []
        pending: Optional[Dict] = None

        for ev in events:
            etype = ev.get("event", "")
            payload = ev.get("payload", {})

            if etype == "training.sample":
                sample = {
                    "prompt": payload.get("context", ""),
                    "completion": json.dumps(payload.get("action", {})),
                    "result": payload.get("result", ""),
                    "step": payload.get("step", 0),
                    "timestamp": ev.get("timestamp", 0),
                }
                samples.append(sample)

        logger.info(f"Extracted {len(samples)} training samples")
        return samples

    def stats(self) -> Dict:
        events = self.load_events()
        sample_events = [e for e in events if e.get("event") == "training.sample"]
        return {
            "total_events": len(events),
            "training_samples": len(sample_events),
            "events_path": str(self.events_path),
        }


class DatasetFormatter:
    """
    Formats extracted samples into standard training formats.
    Outputs: JSONL (default) | Alpaca | ShareGPT
    """

    def __init__(self, output_dir: str = "./data/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_jsonl(self, samples: List[Dict], name: str = "dataset") -> str:
        """Write samples as JSONL. Returns output path."""
        out_path = self.output_dir / f"{name}_{int(time.time())}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                record = {
                    "prompt": s["prompt"],
                    "completion": s["completion"],
                    "metadata": {
                        "step": s.get("step"),
                        "result": s.get("result", "")[:100],
                        "timestamp": s.get("timestamp"),
                    },
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote {len(samples)} samples to {out_path}")
        return str(out_path)

    def to_alpaca(self, samples: List[Dict], name: str = "alpaca") -> str:
        """Alpaca format: instruction / input / output."""
        records = []
        for s in samples:
            records.append({
                "instruction": "You are an autonomous agent. Given the current state, output the correct action.",
                "input": s["prompt"],
                "output": s["completion"],
            })
        out_path = self.output_dir / f"{name}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        return str(out_path)

    def to_sharegpt(self, samples: List[Dict], name: str = "sharegpt") -> str:
        """ShareGPT format: conversations list."""
        conversations = []
        for s in samples:
            conversations.append({
                "conversations": [
                    {"from": "human", "value": s["prompt"]},
                    {"from": "gpt",   "value": s["completion"]},
                ]
            })
        out_path = self.output_dir / f"{name}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(conversations, f, indent=2)
        return str(out_path)

    def summary(self, samples: List[Dict]) -> Dict:
        avg_prompt = sum(len(s["prompt"]) for s in samples) / max(len(samples), 1)
        avg_completion = sum(len(s["completion"]) for s in samples) / max(len(samples), 1)
        return {
            "n_samples": len(samples),
            "avg_prompt_chars": round(avg_prompt),
            "avg_completion_chars": round(avg_completion),
            "estimated_tokens": round((avg_prompt + avg_completion) * len(samples) / 4),
        }
