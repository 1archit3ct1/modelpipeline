"""
state.py — State serializer. Keeps context under token budget.
Critical: never frontload the model with >30k tokens of context.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hard limit: keep context lean. Model throughput > context richness.
MAX_STATE_TOKENS = 15000   # conservative — leaves room for system prompt + response
CHARS_PER_TOKEN  = 4      # rough estimate

# Current observation budget: raw result of immediately preceding action
MAX_OBSERVATION_TOKENS = 4000


def _truncate(text: str, max_tokens: int) -> str:
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


class StateSerializer:
    def __init__(self, path: str = "./data/state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def serialize(
        self,
        task: Optional[Dict] = None,
        recent_steps: Optional[List[Dict]] = None,
        memory_snippets: Optional[List[Dict]] = None,
        agent_vars: Optional[Dict] = None,
        token_budget: int = MAX_STATE_TOKENS,
        current_observation: Optional[str] = None,
    ) -> str:
        """
        Produce a compact state string fit for model context.
        Trims aggressively to stay within token_budget.
        
        current_observation: raw, full result of the immediately preceding action
                           (up to MAX_OBSERVATION_TOKENS) - highest priority after task
        """
        parts = []
        budget = token_budget

        # current task — always included, highest priority
        if task:
            task_str = f"## Current Task\nID: {task.get('id')}\n{task.get('description','')}\nStatus: {task.get('status','')}"
            parts.append(task_str)
            budget -= len(task_str) // CHARS_PER_TOKEN

        # current observation — raw result of last action (critical for agent to see full output)
        if current_observation:
            obs_str = "## Current Observation\n" + _truncate(current_observation, MAX_OBSERVATION_TOKENS)
            parts.append(obs_str)
            budget -= MAX_OBSERVATION_TOKENS

        # recent steps — most recent first, trim if needed
        if recent_steps:
            steps_budget = budget // 2
            steps_str = "## Recent Steps\n"
            for s in reversed(recent_steps[-10:]):
                line = f"- [{s.get('action','?')}] {s.get('summary','')}\n"
                if len(steps_str + line) > steps_budget * CHARS_PER_TOKEN:
                    steps_str += "...(earlier steps trimmed)\n"
                    break
                steps_str += line
            parts.append(steps_str)
            budget -= len(steps_str) // CHARS_PER_TOKEN

        # memory snippets from vector store
        if memory_snippets:
            mem_budget = budget // 2
            mem_str = "## Relevant Memory\n"
            for m in memory_snippets[:5]:
                line = f"- (score={m.get('score',0):.2f}) {m.get('text','')[:200]}\n"
                if len(mem_str + line) > mem_budget * CHARS_PER_TOKEN:
                    break
                mem_str += line
            parts.append(mem_str)

        # agent vars — small key/value pairs
        if agent_vars:
            var_str = "## Agent State\n" + json.dumps(agent_vars, indent=2)
            var_str = _truncate(var_str, budget // 4)
            parts.append(var_str)

        state = "\n\n".join(parts)
        return _truncate(state, token_budget)

    def save(self, state: Dict[str, Any]):
        from src.agent.hooks import emit, Event
        state["saved_at"] = time.time()
        with open(self.path, "w") as f:
            json.dump(state, f, indent=2)
        emit(Event.STATE_SAVED, {"path": str(self.path)})

    def load(self) -> Optional[Dict[str, Any]]:
        from src.agent.hooks import emit, Event
        if not self.path.exists():
            return None
        with open(self.path) as f:
            state = json.load(f)
        emit(Event.STATE_LOADED, {"path": str(self.path)})
        return state
