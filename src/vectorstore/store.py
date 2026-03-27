"""
store.py — In-house vector store. HNSW index + numpy array + jsonlines metadata.
No Chroma. No FAISS black box. You own every byte.
"""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.jsonl"
INDEX_FILE = "index.bin"
STATS_FILE = "stats.json"


class VectorStore:
    """
    Thread-safe vector store backed by:
    - numpy float32 array (vectors.npy)
    - jsonlines metadata (metadata.jsonl)
    - hnswlib HNSW index (index.bin) for ANN search
    """

    def __init__(self, path: str = "./data/memory", dim: int = 768):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()

        self._vectors: Optional[np.ndarray] = None   # shape (N, dim)
        self._metadata: List[Dict[str, Any]] = []
        self._index = None  # hnswlib index
        self._dirty = False

        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        vpath = self.path / VECTORS_FILE
        mpath = self.path / METADATA_FILE

        if vpath.exists():
            self._vectors = np.load(str(vpath))
            logger.info(f"Loaded {len(self._vectors)} vectors from {vpath}")
        else:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)

        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                self._metadata = [json.loads(l) for l in f if l.strip()]

        self._rebuild_index()

    def _save(self):
        np.save(str(self.path / VECTORS_FILE), self._vectors)
        with open(self.path / METADATA_FILE, "w", encoding="utf-8") as f:
            for m in self._metadata:
                f.write(json.dumps(m) + "\n")
        self._save_index()
        self._write_stats()
        self._dirty = False

    def _save_index(self):
        if self._index is not None:
            self._index.save_index(str(self.path / INDEX_FILE))

    def _rebuild_index(self):
        n = len(self._vectors)
        if n == 0:
            self._index = None
            return
        try:
            import hnswlib
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            # M=16 ef_construction=200 — balanced for <1M vectors
            idx.init_index(max_elements=max(n * 2, 1000), M=16, ef_construction=200)
            idx.add_items(self._vectors, list(range(n)))
            idx.set_ef(50)
            self._index = idx
            logger.debug(f"HNSW index rebuilt with {n} vectors")
        except ImportError:
            logger.warning("hnswlib not installed — falling back to brute-force cosine")
            self._index = None

    def _write_stats(self):
        stats = {
            "count": len(self._vectors),
            "dim": self.dim,
            "size_bytes": self._vectors.nbytes if self._vectors is not None else 0,
            "updated_at": time.time(),
        }
        with open(self.path / STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)

    # ── public API ────────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[np.ndarray] = None,
    ) -> int:
        """
        Store a text (and optional precomputed vector) in the store.
        Returns the assigned integer ID.
        """
        from src.vectorstore.embedder import embed

        if vector is None:
            vector = embed(text)[0]

        vector = np.array(vector, dtype=np.float32).reshape(1, self.dim)

        with self._lock:
            idx = len(self._vectors)
            self._vectors = np.vstack([self._vectors, vector])

            entry = {
                "id": idx,
                "text": text,
                "timestamp": time.time(),
                **(metadata or {}),
            }
            self._metadata.append(entry)

            # add to live index without full rebuild
            if self._index is not None:
                try:
                    if self._index.get_current_count() >= self._index.get_max_elements():
                        self._rebuild_index()
                    else:
                        self._index.add_items(vector, [idx])
                except Exception:
                    self._rebuild_index()
            else:
                self._rebuild_index()

            self._dirty = True
            self._save()
            return idx

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_vector: Optional[np.ndarray] = None,
        log_retrieval: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar entries to query.
        Returns list of metadata dicts with added 'score' field.
        """
        from src.vectorstore.embedder import embed_query

        if len(self._vectors) == 0:
            return []

        if query_vector is None:
            query_vector = embed_query(query)

        query_vector = np.array(query_vector, dtype=np.float32)

        with self._lock:
            if self._index is not None:
                k = min(top_k, len(self._vectors))
                labels, distances = self._index.knn_query(
                    query_vector.reshape(1, -1), k=k
                )
                # hnswlib cosine distance = 1 - similarity
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    entry = dict(self._metadata[label])
                    entry["score"] = float(1.0 - dist)
                    results.append(entry)
            else:
                # brute-force fallback
                scores = self._vectors @ query_vector
                top = np.argsort(scores)[::-1][:top_k]
                results = []
                for i in top:
                    entry = dict(self._metadata[i])
                    entry["score"] = float(scores[i])
                    results.append(entry)

        if log_retrieval:
            self._log_retrieval(query, results)

        return results

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._vectors),
                "dim": self.dim,
                "size_mb": round(self._vectors.nbytes / 1e6, 3),
                "index_type": "hnsw" if self._index is not None else "brute_force",
                "path": str(self.path),
            }

    def clear(self):
        with self._lock:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._metadata = []
            self._index = None
            self._save()

    # ── training data hook ────────────────────────────────────────────────────

    def _log_retrieval(self, query: str, results: List[Dict]):
        """
        Every retrieval is logged for training data capture.
        Format: {query, results, timestamp} → appended to retrieval_log.jsonl
        """
        log_path = self.path / "retrieval_log.jsonl"
        entry = {
            "query": query,
            "results": [{"id": r["id"], "score": r["score"], "text": r.get("text", "")} for r in results],
            "timestamp": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
