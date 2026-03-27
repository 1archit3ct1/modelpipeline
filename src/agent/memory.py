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

    def index_workspace(
        self,
        workspace: str = "./workspace",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Automatically chunk and embed all source files in workspace into VectorStore.

        Args:
            workspace: Path to workspace directory
            chunk_size: Characters per chunk (default: 500)
            chunk_overlap: Overlap between chunks (default: 50)
            extensions: File extensions to index (default: [".py", ".md"])

        Returns:
            Dict with stats: {"files_processed": N, "chunks_created": M}
        """
        from src.agent.hooks import emit, Event

        if extensions is None:
            extensions = [".py", ".md"]

        workspace_path = Path(workspace)
        if not workspace_path.exists():
            return {"files_processed": 0, "chunks_created": 0, "error": "workspace not found"}

        files_processed = 0
        chunks_created = 0

        for ext in extensions:
            for file_path in workspace_path.rglob(f"*{ext}"):
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    if not content.strip():
                        continue

                    # Chunk the file content
                    chunks = self._chunk_text(
                        content,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        file_path=str(file_path.relative_to(workspace_path)),
                    )

                    for chunk in chunks:
                        self.store(
                            text=chunk["text"],
                            metadata={
                                "type": "file_chunk",
                                "file": chunk["file_path"],
                                "chunk_index": chunk["chunk_index"],
                                "total_chunks": chunk["total_chunks"],
                                "extension": ext,
                            },
                        )
                        chunks_created += 1

                    files_processed += 1

                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")

        result = {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
        }

        emit(Event.MEMORY_INDEXED, result)
        return result

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        file_path: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Attempts to split on line boundaries for cleaner chunks.
        """
        if len(text) <= chunk_size:
            return [
                {
                    "text": text,
                    "file_path": file_path,
                    "chunk_index": 0,
                    "total_chunks": 1,
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # If not at end of text, try to break at newline
            if end < len(text):
                # Look for last newline within chunk
                last_newline = text.rfind("\n", start, end)
                if last_newline > start:
                    end = last_newline + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "total_chunks": -1,  # will update at end
                    }
                )
                chunk_index += 1

            # Move start with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

        # Update total_chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks
