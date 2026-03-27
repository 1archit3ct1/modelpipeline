#!/usr/bin/env python
"""
Test script to verify the advanced vector retrieval improvements.
Tests MMR, hybrid scoring, score normalization, and adaptive search.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.vectorstore.store import VectorStore


def test_mmr_retrieval():
    """Test Maximal Marginal Relevance reranking for diversity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768, default_mmr_lambda=0.7)
        
        # Store similar documents about Python programming
        docs = [
            "Python is a programming language for data science",
            "Python programming for machine learning applications",
            "Python code for web development frameworks",
            "Java is a programming language for enterprise software",
            "JavaScript for frontend web development",
        ]
        
        for doc in docs:
            store.store(doc)
        
        # Query for Python programming
        results_cosine = store.retrieve("Python programming", top_k=3, strategy="cosine")
        results_mmr = store.retrieve("Python programming", top_k=3, strategy="mmr", mmr_lambda=0.7)
        
        assert len(results_cosine) == 3, "Should return 3 cosine results"
        assert len(results_mmr) == 3, "Should return 3 MMR results"
        
        # MMR results should have strategy marker
        for r in results_mmr:
            assert r.get("strategy") == "mmr", "MMR results should have strategy marker"
            assert "mmr_lambda" in r, "MMR results should have lambda value"
        
        print("✓ MMR retrieval works correctly")
        return True


def test_score_normalization():
    """Test z-score normalization of retrieval scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768)
        
        # Store diverse documents
        docs = [
            "Machine learning with neural networks",
            "Deep learning for computer vision",
            "Natural language processing transformers",
            "Reinforcement learning algorithms",
            "Statistical analysis methods",
        ]
        
        for doc in docs:
            store.store(doc)
        
        # Multiple queries to build score history
        for query in ["machine learning", "deep learning", "NLP"]:
            results = store.retrieve(query, top_k=3, use_normalization=True)
            
            for r in results:
                assert "normalized_score" in r, "Should have normalized score"
                assert "score_mean" in r, "Should have score mean"
                assert "score_std" in r, "Should have score std"
        
        print("✓ Score normalization works correctly")
        return True


def test_adaptive_ef_search():
    """Test adaptive HNSW ef parameter based on query characteristics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768)
        
        # Store documents
        for i in range(20):
            store.store(f"Document {i} about topic {i % 5}")
        
        # Short query should get higher ef
        query_info_short = store._analyze_query("ML", store._vectors[0])
        ef_short = store._compute_adaptive_ef(query_info_short, top_k=5, candidate_factor=3)
        
        # Long specific query should get lower ef
        long_query = "machine learning neural network deep learning transformer"
        query_info_long = store._analyze_query(long_query, store._vectors[0])
        ef_long = store._compute_adaptive_ef(query_info_long, top_k=5, candidate_factor=3)
        
        assert query_info_short["is_short"], "Short query should be detected"
        assert ef_short >= ef_long, "Short queries should get higher ef"
        
        print(f"✓ Adaptive ef: short={ef_short}, long={ef_long}")
        return True


def test_hybrid_scoring():
    """Test hybrid semantic + lexical scoring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768, enable_hybrid=True)
        
        # Store documents with overlapping keywords
        store.store("Python programming language tutorial")
        store.store("Java programming enterprise guide")
        store.store("Python snake reptile information")
        
        # Query with specific terms
        results = store.retrieve(
            "Python programming",
            top_k=3,
            strategy="hybrid",
            use_normalization=False,
        )
        
        assert len(results) > 0, "Should return results"
        
        # Check hybrid score components
        for r in results:
            assert "hybrid_score" in r or r.get("strategy") != "hybrid", \
                "Hybrid results should have hybrid_score"
        
        print("✓ Hybrid scoring works correctly")
        return True


def test_query_analysis():
    """Test query characteristic analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768)
        store.store("test document")
        
        # Analyze different query types
        info_short = store._analyze_query("hello", store._vectors[0])
        info_long = store._analyze_query(
            "machine learning deep neural network transformer architecture",
            store._vectors[0]
        )
        info_specific = store._analyze_query(
            "how to implement JWT authentication in Flask API",
            store._vectors[0]
        )
        
        assert info_short["is_short"], "Short query should be detected"
        assert not info_long["is_short"], "Long query should not be short"
        assert info_specific["term_count"] > 5, "Specific query has many terms"
        
        print("✓ Query analysis works correctly")
        return True


def test_detailed_logging():
    """Test detailed retrieval logging for training data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768)
        store.store("test content for logging")
        
        # Retrieve with logging
        store.retrieve("test query", top_k=3, strategy="mmr", mmr_lambda=0.5)
        
        # Check log file exists and has content
        log_path = Path(tmpdir) / "retrieval_log.jsonl"
        assert log_path.exists(), "Log file should exist"
        
        with open(log_path) as f:
            log_entry = __import__('json').loads(f.readline())
        
        assert "query_info" in log_entry, "Should log query info"
        assert "strategy" in log_entry, "Should log strategy"
        assert "mmr_lambda" in log_entry, "Should log MMR lambda"
        assert "score_stats" in log_entry, "Should log score stats"
        
        print("✓ Detailed logging works correctly")
        return True


def test_backward_compatibility():
    """Test that old retrieve() calls still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(path=tmpdir, dim=768)
        store.store("backward compatibility test")
        
        # Old-style call without new parameters
        results = store.retrieve("test", top_k=5)
        
        assert len(results) <= 5, "Should return up to 5 results"
        assert results[0]["score"] is not None, "Should have score"
        
        print("✓ Backward compatibility maintained")
        return True


if __name__ == "__main__":
    print("Testing Advanced Vector Retrieval Improvements...\n")
    
    tests = [
        test_mmr_retrieval,
        test_score_normalization,
        test_adaptive_ef_search,
        test_hybrid_scoring,
        test_query_analysis,
        test_detailed_logging,
        test_backward_compatibility,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}\n")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}\n")
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
