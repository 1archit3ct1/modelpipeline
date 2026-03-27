"""
memory.py — Two-tier memory: fast KV store (short-term) + vector store (long-term).
No-context resume: agent can pick up exactly where it left off.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        kv_path: str = "./data/memory/kv.json",
        vector_path: str = "./data/memory/vectors",
    ):
        self.kv_path = Path(kv_path)
        self.kv_path.parent.mkdir(parents=True, exist_ok=True)
        self._kv: Dict[str, Any] = {}
        self._load_kv()

        from src.vectorstore.store import VectorStore
        self._vector = VectorStore(path=vector_path)

    # ── KV (short-term, exact lookup) ────────────────────────────────────────

    def _load_kv(self):
        if self.kv_path.exists():
            with open(self.kv_path) as f:
                self._kv = json.load(f)

    def _save_kv(self):
        with open(self.kv_path, "w") as f:
            json.dump(self._kv, f, indent=2)

    def set(self, key: str, value: Any):
        self._kv[key] = {"value": value, "updated_at": time.time()}
        self._save_kv()

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._kv.get(key)
        return entry["value"] if entry else default

    def delete(self, key: str):
        self._kv.pop(key, None)
        self._save_kv()

    # ── Vector (long-term, semantic search) ──────────────────────────────────

    def store(self, text: str, metadata: Optional[Dict] = None) -> int:
        from src.agent.hooks import emit, Event
        idx = self._vector.store(text, metadata)
        emit(Event.MEMORY_STORED, {"id": idx, "text": text[:100]})
        return idx

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        from src.agent.hooks import emit, Event
        results = self._vector.retrieve(query, top_k=top_k)
        emit(Event.MEMORY_RETRIEVED, {"query": query, "n_results": len(results)})
        return results

    # ── Resume context ───────────────────────────────────────────────────────

    def resume_context(self, task_description: str) -> Dict[str, Any]:
        """
        Build a minimal resume context for a task.
        Returns KV snapshot + top memory hits.
        Keeps context lean — no frontloading.
        """
        snippets = self.retrieve(task_description, top_k=3)
        return {
            "kv_snapshot": {k: v["value"] for k, v in list(self._kv.items())[-20:]},
            "memory_snippets": snippets,
        }

    def stats(self) -> Dict:
        return {
            "kv_keys": len(self._kv),
            "vector": self._vector.stats(),
        }
