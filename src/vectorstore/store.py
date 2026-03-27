"""
store.py — In-house vector store with hybrid retrieval and MMR diversification.

ARCHITECTURE:
- numpy float32 array (vectors.npy) — dense embeddings
- jsonlines metadata (metadata.jsonl) — document metadata
- hnswlib HNSW index (index.bin) — approximate nearest neighbor search

RETRIEVAL PIPELINE:
1. Query analysis — detect query type (short/long, specific/broad)
2. Adaptive search — adjust HNSW ef parameter based on query characteristics
3. Score normalization — z-score calibration across queries
4. MMR diversification — reduce redundancy in top-k results
5. Hybrid scoring — combine semantic + lexical signals (optional)

TRAINING DATA CAPTURE:
Every retrieval logs: query vector stats, raw scores, normalized scores,
MMR lambda, final ranking. This enables offline analysis of what works.
"""

from __future__ import annotations
import os
import json
import time
import logging
import threading
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.jsonl"
INDEX_FILE = "index.bin"
STATS_FILE = "stats.json"
RETRIEVAL_LOG_FILE = "retrieval_log.jsonl"


class VectorStore:
    """
    Thread-safe vector store with advanced retrieval strategies.

    RETRIEVAL STRATEGIES:
    - cosine: Standard cosine similarity (default, fast)
    - mmr: Maximal Marginal Relevance (diverse results)
    - hybrid: Combines semantic + lexical overlap scores

    MATHEMATICAL FOUNDATIONS:

    1. Cosine Similarity:
       sim(A,B) = (A · B) / (||A|| × ||B||)
       Range: [-1, 1], higher = more similar

    2. MMR (Maximal Marginal Relevance):
       MMR = argmax [ λ × sim(doc, query) - (1-λ) × max_sim(doc, selected) ]
       λ controls diversity vs relevance tradeoff
       λ=1.0 → pure relevance, λ=0.0 → pure diversity

    3. Z-Score Normalization:
       z = (score - μ) / σ
       Calibrates scores across different query distributions
    """

    def __init__(
        self,
        path: str = "./data/memory",
        dim: int = 768,
        default_mmr_lambda: float = 0.7,
        enable_hybrid: bool = False,
    ):
        """
        Initialize vector store.

        Args:
            path: Directory for persistence files
            dim: Embedding dimension (default: 768 for nomic-embed-text)
            default_mmr_lambda: MMR diversity parameter (0=diverse, 1=relevant)
            enable_hybrid: Enable hybrid semantic+lexical scoring
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.default_mmr_lambda = default_mmr_lambda
        self.enable_hybrid = enable_hybrid

        self._lock = threading.RLock()
        self._vectors: Optional[np.ndarray] = None   # shape (N, dim)
        self._metadata: List[Dict[str, Any]] = []
        self._index = None  # hnswlib index
        self._dirty = False

        # Score statistics for normalization
        self._score_history: List[float] = []
        self._score_mean: float = 0.0
        self._score_std: float = 1.0

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
        strategy: str = "cosine",
        mmr_lambda: Optional[float] = None,
        use_normalization: bool = True,
        candidate_factor: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar entries to query with advanced routing.

        RETRIEVAL STRATEGIES:
        - "cosine": Standard cosine similarity (fastest, default)
        - "mmr": Maximal Marginal Relevance (diverse results)
        - "hybrid": Combines semantic + lexical overlap

        Args:
            query: Search query string
            top_k: Number of results to return
            query_vector: Optional pre-computed query embedding
            log_retrieval: Whether to log for training data
            strategy: Retrieval strategy ("cosine", "mmr", "hybrid")
            mmr_lambda: MMR diversity parameter (0=diverse, 1=relevant), default 0.7
            use_normalization: Apply z-score normalization to scores
            candidate_factor: Fetch top_k * candidate_factor for MMR reranking

        Returns:
            List of metadata dicts with 'score' and 'normalized_score' fields

        ROUTING LOGIC:
        1. Analyze query (length, specificity)
        2. Compute adaptive ef for HNSW search
        3. Fetch candidate pool (top_k * candidate_factor)
        4. Apply MMR reranking if strategy="mmr"
        5. Normalize scores using running statistics
        6. Return top_k results
        """
        from src.vectorstore.embedder import embed_query

        if len(self._vectors) == 0:
            return []

        if query_vector is None:
            query_vector = embed_query(query)

        query_vector = np.array(query_vector, dtype=np.float32)

        # Analyze query for adaptive search
        query_info = self._analyze_query(query, query_vector)
        adaptive_ef = self._compute_adaptive_ef(query_info, top_k, candidate_factor)

        with self._lock:
            # Fetch candidate pool
            raw_results = self._fetch_candidates(
                query_vector, top_k, candidate_factor, adaptive_ef
            )

            # Apply retrieval strategy
            if strategy == "mmr" or (strategy == "cosine" and mmr_lambda is not None):
                results = self._apply_mmr(
                    query_vector, raw_results, top_k, mmr_lambda or self.default_mmr_lambda
                )
            elif strategy == "hybrid" and self.enable_hybrid:
                results = self._apply_hybrid_scoring(query, raw_results, top_k)
            else:
                results = raw_results[:top_k]

            # Normalize scores
            if use_normalization:
                results = self._normalize_scores(results)

        if log_retrieval:
            self._log_retrieval_detailed(
                query, results, query_info, strategy, mmr_lambda or self.default_mmr_lambda
            )

        return results

    # ── Query Analysis ───────────────────────────────────────────────────────

    def _analyze_query(self, query: str, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        Analyze query characteristics to inform retrieval strategy.

        Captures:
        - Query length (short queries may need broader search)
        - Term count (specific vs broad)
        - Vector magnitude (confidence indicator)
        - Entropy estimate (ambiguity)
        """
        words = query.split()
        term_count = len(words)

        # Vector magnitude indicates embedding confidence
        magnitude = float(np.linalg.norm(query_vector))

        # Estimate specificity: more unique terms = more specific
        unique_terms = len(set(words))
        specificity = unique_terms / max(term_count, 1)

        return {
            "query_length": len(query),
            "term_count": term_count,
            "unique_terms": unique_terms,
            "specificity": specificity,
            "vector_magnitude": magnitude,
            "is_short": len(query) < 20,
            "is_specific": specificity > 0.8 and term_count > 3,
        }

    def _compute_adaptive_ef(
        self,
        query_info: Dict[str, Any],
        top_k: int,
        candidate_factor: int,
    ) -> int:
        """
        Compute adaptive HNSW ef parameter based on query characteristics.

        RATIONALE:
        - Short/ambiguous queries need broader search (higher ef)
        - Specific queries can use narrower search (lower ef)
        - Minimum ef = top_k * candidate_factor

        Returns:
            ef parameter for HNSW search
        """
        base_ef = top_k * candidate_factor

        # Increase ef for short/ambiguous queries
        if query_info["is_short"]:
            base_ef = int(base_ef * 1.5)

        # Decrease ef for highly specific queries
        if query_info["is_specific"]:
            base_ef = int(base_ef * 0.8)

        # Clamp to reasonable bounds
        return max(min(base_ef, 500), max(50, base_ef))

    # ── Candidate Fetching ───────────────────────────────────────────────────

    def _fetch_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int,
        candidate_factor: int,
        ef: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch candidate pool from HNSW index or brute-force fallback.

        Returns more candidates than needed for MMR reranking.
        """
        k = min(top_k * candidate_factor, len(self._vectors))

        if self._index is not None:
            # Set adaptive ef for this query
            self._index.set_ef(ef)

            labels, distances = self._index.knn_query(
                query_vector.reshape(1, -1), k=k
            )

            results = []
            for label, dist in zip(labels[0], distances[0]):
                entry = dict(self._metadata[label])
                entry["score"] = float(1.0 - dist)  # cosine similarity
                entry["raw_distance"] = float(dist)
                results.append(entry)
        else:
            # Brute-force fallback
            scores = self._vectors @ query_vector
            top_indices = np.argsort(scores)[::-1][:k]

            results = []
            for i in top_indices:
                entry = dict(self._metadata[i])
                entry["score"] = float(scores[i])
                entry["raw_distance"] = float(1.0 - scores[i])
                results.append(entry)

        return results

    # ── MMR Reranking ────────────────────────────────────────────────────────

    def _apply_mmr(
        self,
        query_vector: np.ndarray,
        candidates: List[Dict[str, Any]],
        top_k: int,
        mmr_lambda: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance reranking for diversity.

        ALGORITHM:
        selected = []
        while len(selected) < top_k:
            best = argmax [ λ × sim(doc, query) - (1-λ) × max_sim(doc, selected) ]
            selected.append(best)

        INTUITION:
        - First pick: highest similarity to query
        - Subsequent picks: balance relevance vs novelty
        - λ=1.0: pure relevance ranking
        - λ=0.0: pure diversity (maximally different from each other)
        """
        if not candidates or top_k <= 0:
            return candidates[:top_k]

        selected = []
        remaining = list(candidates)

        # Precompute pairwise similarities for efficiency
        candidate_vectors = []
        for c in candidates:
            # Reconstruct vector from metadata or compute on fly
            idx = c.get("id", 0)
            if idx < len(self._vectors):
                candidate_vectors.append(self._vectors[idx])
            else:
                candidate_vectors.append(query_vector)  # fallback

        while len(selected) < min(top_k, len(candidates)) and remaining:
            best_score = float("-inf")
            best_idx = -1

            for i, candidate in enumerate(remaining):
                # Relevance to query
                relevance = candidate["score"]

                # Redundancy with already selected
                max_sim_to_selected = 0.0
                for sel_idx in selected:
                    sim = self._cosine_sim_vectors(
                        candidate_vectors[i], candidate_vectors[sel_idx]
                    )
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.index(remaining[best_idx]))
                remaining.pop(best_idx)
                candidate_vectors.pop(best_idx)

        # Return in original order with MMR indicator
        results = [candidates[i] for i in selected[:top_k]]
        for r in results:
            r["strategy"] = "mmr"
            r["mmr_lambda"] = mmr_lambda

        return results

    def _cosine_sim_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # ── Hybrid Scoring ───────────────────────────────────────────────────────

    def _apply_hybrid_scoring(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring combining semantic + lexical signals.

        SCORE = α × semantic_score + (1-α) × lexical_score

        Lexical score uses Jaccard similarity on word sets.
        """
        query_words = set(query.lower().split())

        for candidate in candidates:
            doc_text = candidate.get("text", "").lower()
            doc_words = set(re.findall(r"\w+", doc_text))

            # Jaccard similarity
            if query_words and doc_words:
                intersection = len(query_words & doc_words)
                union = len(query_words | doc_words)
                lexical_score = intersection / union
            else:
                lexical_score = 0.0

            # Combine scores
            semantic = candidate["score"]
            candidate["hybrid_score"] = alpha * semantic + (1 - alpha) * lexical_score
            candidate["lexical_score"] = lexical_score

        # Sort by hybrid score
        candidates.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return candidates[:top_k]

    # ── Score Normalization ──────────────────────────────────────────────────

    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        decay: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Apply z-score normalization to scores using running statistics.

        RATIONALE:
        - Raw cosine scores vary across queries
        - Normalization enables cross-query comparison
        - Running stats adapt to corpus over time
        """
        if not results:
            return results

        scores = [r["score"] for r in results]

        # Update running statistics with exponential decay
        current_mean = np.mean(scores)
        current_std = np.std(scores) + 1e-9  # avoid division by zero

        self._score_mean = decay * self._score_mean + (1 - decay) * current_mean
        self._score_std = decay * self._score_std + (1 - decay) * current_std

        # Z-score normalization
        for r in results:
            z_score = (r["score"] - self._score_mean) / self._score_std
            r["normalized_score"] = float(z_score)
            r["score_mean"] = self._score_mean
            r["score_std"] = self._score_std

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
        """Legacy method for backward compatibility."""
        self._log_retrieval_detailed(query, results, {}, "cosine", 1.0)

    def _log_retrieval_detailed(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        strategy: str,
        mmr_lambda: float,
    ):
        """
        Log retrieval with full metadata for training data analysis.

        CAPTURED DATA:
        - Query characteristics (length, specificity, vector magnitude)
        - Strategy used (cosine, mmr, hybrid)
        - MMR lambda parameter
        - Raw and normalized scores
        - Final ranking order

        This enables offline analysis of:
        - Which strategies work best for different query types
        - Optimal MMR lambda values for different use cases
        - Score distribution patterns across queries
        """
        log_path = self.path / RETRIEVAL_LOG_FILE
        entry = {
            "query": query,
            "query_info": query_info,
            "strategy": strategy,
            "mmr_lambda": mmr_lambda,
            "results": [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "normalized_score": r.get("normalized_score"),
                    "text": r.get("text", "")[:200],  # truncate for log size
                    "strategy": r.get("strategy"),
                }
                for r in results
            ],
            "score_stats": {
                "mean": self._score_mean,
                "std": self._score_std,
            },
            "timestamp": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
