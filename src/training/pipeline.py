"""
training/pipeline.py — Trajectory capture and dataset formatter.
Reads events.jsonl → formats into training pairs → outputs JSONL dataset.
This IS the data flywheel. Every run improves the next model.

FEATURES:
- Quality scoring: Filter samples by action success, task completion, result quality
- Episode boundaries: Group samples by task_id for episodic training
- DPO pairs: Extract (chosen, rejected) pairs for Direct Preference Optimization
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

    def extract_samples(self, min_quality_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Extract training samples from event log with optional quality filtering.

        Args:
            min_quality_score: Filter samples below this quality threshold (0.0-1.0)

        Each sample: {prompt, completion, result, metadata, quality_score}
        """
        events = self.load_events()
        samples = []

        for ev in events:
            etype = ev.get("event", "")
            payload = ev.get("payload", {})

            if etype == "training.sample":
                quality_signals = payload.get("quality_signals", {})
                quality_score = self._compute_quality_score(quality_signals)

                if quality_score < min_quality_score:
                    continue

                sample = {
                    "prompt": payload.get("context", ""),
                    "completion": json.dumps(payload.get("action", {})),
                    "result": payload.get("result", ""),
                    "step": payload.get("step", 0),
                    "timestamp": ev.get("timestamp", 0),
                    "quality_score": quality_score,
                    "quality_signals": quality_signals,
                }
                samples.append(sample)

        logger.info(f"Extracted {len(samples)} training samples (quality >= {min_quality_score})")
        return samples

    def _compute_quality_score(self, signals: Dict[str, Any]) -> float:
        """
        Compute quality score from signals (0.0-1.0).

        Scoring:
        - action_succeeded: +0.3
        - task_completed: +0.3
        - result_length > 100: +0.1 (substantial output)
        - not terminal failure: +0.2
        - step efficiency (fewer steps = higher): +0.1
        """
        score = 0.0

        if signals.get("action_succeeded", False):
            score += 0.3

        if signals.get("task_status") == "completed":
            score += 0.3

        if signals.get("result_length", 0) > 100:
            score += 0.1

        if not signals.get("is_terminal_action", False) or signals.get("task_status") == "completed":
            score += 0.2

        # Step efficiency: early steps often more valuable
        step = signals.get("step_in_task", 1)
        if step <= 5:
            score += 0.1

        return min(score, 1.0)

    def extract_episodes(self) -> List[Dict[str, Any]]:
        """
        Extract complete task episodes for episodic training.

        Returns list of episodes:
        {
            "task_id": str,
            "task_description": str,
            "samples": [...],
            "success": bool,
            "total_steps": int,
            "quality_score": float (avg),
        }
        """
        events = self.load_events()

        # Group by task (using task.created events as boundaries)
        episodes = {}
        current_task_id = None

        for ev in events:
            etype = ev.get("event", "")
            payload = ev.get("payload", {})

            if etype == "task.created":
                current_task_id = payload.get("task_id", str(len(episodes)))
                episodes[current_task_id] = {
                    "task_id": current_task_id,
                    "task_description": payload.get("description", ""),
                    "samples": [],
                    "success": False,
                    "total_steps": 0,
                }

            elif etype == "task.completed" and current_task_id:
                episodes[current_task_id]["success"] = True
                episodes[current_task_id]["task_result"] = payload.get("result", "")

            elif etype == "task.failed" and current_task_id:
                episodes[current_task_id]["success"] = False
                episodes[current_task_id]["task_error"] = payload.get("error", "")

            elif etype == "training.sample" and current_task_id:
                quality_signals = payload.get("quality_signals", {})
                quality_score = self._compute_quality_score(quality_signals)

                episodes[current_task_id]["samples"].append({
                    "step": payload.get("step", 0),
                    "prompt": payload.get("context", ""),
                    "completion": json.dumps(payload.get("action", {})),
                    "result": payload.get("result", ""),
                    "quality_score": quality_score,
                })
                episodes[current_task_id]["total_steps"] = max(
                    episodes[current_task_id]["total_steps"],
                    payload.get("step", 0)
                )

        # Compute average quality per episode
        for ep in episodes.values():
            if ep["samples"]:
                ep["quality_score"] = sum(s["quality_score"] for s in ep["samples"]) / len(ep["samples"])
            else:
                ep["quality_score"] = 0.0

        return list(episodes.values())

    def extract_dpo_pairs(self) -> List[Dict[str, Any]]:
        """
        Extract Direct Preference Optimization pairs.

        Finds steps where:
        1. First attempt failed (rejected)
        2. Subsequent correction succeeded (chosen)

        Returns:
        {
            "prompt": str,
            "chosen": str (successful action),
            "rejected": str (failed action),
            "reason": str (why chosen is better),
        }
        """
        episodes = self.extract_episodes()
        dpo_pairs = []

        for ep in episodes:
            samples = sorted(ep["samples"], key=lambda s: s["step"])

            for i, sample in enumerate(samples):
                if sample["quality_score"] < 0.5:  # Low quality = rejected
                    # Look for subsequent higher-quality sample with similar context
                    for j in range(i + 1, min(i + 3, len(samples))):
                        if samples[j]["quality_score"] > 0.7:  # High quality = chosen
                            dpo_pairs.append({
                                "prompt": sample["prompt"][:2000],  # Truncate for DPO format
                                "chosen": samples[j]["completion"],
                                "rejected": sample["completion"],
                                "reason": f"Step {samples[j]['step']} succeeded where step {sample['step']} failed",
                                "task_id": ep["task_id"],
                            })
                            break

        logger.info(f"Extracted {len(dpo_pairs)} DPO pairs")
        return dpo_pairs

    def stats(self) -> Dict:
        events = self.load_events()
        sample_events = [e for e in events if e.get("event") == "training.sample"]

        # Compute quality distribution
        quality_scores = []
        for ev in sample_events:
            signals = ev.get("payload", {}).get("quality_signals", {})
            quality_scores.append(self._compute_quality_score(signals))

        return {
            "total_events": len(events),
            "training_samples": len(sample_events),
            "avg_quality_score": round(sum(quality_scores) / max(len(quality_scores), 1), 3),
            "high_quality_samples": len([s for s in quality_scores if s > 0.7]),
            "events_path": str(self.events_path),
        }


class DatasetFormatter:
    """
    Formats extracted samples into standard training formats.
    Outputs: JSONL (default) | Alpaca | ShareGPT | DPO
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
                        "quality_score": s.get("quality_score"),
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
                "quality_score": s.get("quality_score", 0.5),
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
                ],
                "metadata": {
                    "quality_score": s.get("quality_score", 0.5),
                    "step": s.get("step"),
                }
            })
        out_path = self.output_dir / f"{name}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(conversations, f, indent=2)
        return str(out_path)

    def to_dpo(self, dpo_pairs: List[Dict], name: str = "dpo") -> str:
        """
        DPO (Direct Preference Optimization) format.
        Each entry: {prompt, chosen, rejected}
        """
        out_path = self.output_dir / f"{name}_{int(time.time())}.jsonl"
        with open(out_path, "w") as f:
            for pair in dpo_pairs:
                record = {
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                    "reason": pair.get("reason", ""),
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote {len(dpo_pairs)} DPO pairs to {out_path}")
        return str(out_path)

    def to_episodic(self, episodes: List[Dict], name: str = "episodic") -> str:
        """
        Episodic format: complete task episodes with all steps.
        """
        out_path = self.output_dir / f"{name}_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(episodes, f, indent=2)
        logger.info(f"Wrote {len(episodes)} episodes to {out_path}")
        return str(out_path)

    def summary(self, samples: List[Dict]) -> Dict:
        avg_prompt = sum(len(s["prompt"]) for s in samples) / max(len(samples), 1)
        avg_completion = sum(len(s["completion"]) for s in samples) / max(len(samples), 1)
        avg_quality = sum(s.get("quality_score", 0.5) for s in samples) / max(len(samples), 1)
        return {
            "n_samples": len(samples),
            "avg_prompt_chars": round(avg_prompt),
            "avg_completion_chars": round(avg_completion),
            "avg_quality_score": round(avg_quality, 3),
            "estimated_tokens": round((avg_prompt + avg_completion) * len(samples) / 4),
        }
