#!/usr/bin/env python
"""
Test script to verify the memory.index_workspace functionality.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.agent.memory import Memory
from src.agent.actions import ActionType, parse_and_validate


def test_action_type_exists():
    """Verify MEMORY_INDEX_WORKSPACE action type is defined."""
    assert hasattr(ActionType, "MEMORY_INDEX_WORKSPACE"), "MEMORY_INDEX_WORKSPACE action type should exist"
    assert ActionType.MEMORY_INDEX_WORKSPACE.value == "memory.index_workspace", "Should have correct value"
    print("✓ MEMORY_INDEX_WORKSPACE action type is defined")
    return True


def test_action_required_params():
    """Verify MEMORY_INDEX_WORKSPACE has no required params (all optional)."""
    from src.agent.actions import REQUIRED_PARAMS
    
    assert ActionType.MEMORY_INDEX_WORKSPACE in REQUIRED_PARAMS, "MEMORY_INDEX_WORKSPACE should have required params"
    assert REQUIRED_PARAMS[ActionType.MEMORY_INDEX_WORKSPACE] == [], "All params should be optional"
    
    print("✓ MEMORY_INDEX_WORKSPACE has correct required params (none)")
    return True


def test_action_parsing():
    """Verify memory.index_workspace action parses correctly."""
    # Test with no params
    json_input = '{"action": "memory.index_workspace"}'
    actions = parse_and_validate(json_input)
    
    assert len(actions) == 1, "Should parse one action"
    assert actions[0].type == ActionType.MEMORY_INDEX_WORKSPACE, "Should be MEMORY_INDEX_WORKSPACE type"
    assert actions[0].valid, "Should be valid"
    
    # Test with optional params
    json_input2 = '{"action": "memory.index_workspace", "chunk_size": 300, "extensions": [".py"]}'
    actions2 = parse_and_validate(json_input2)
    
    assert len(actions2) == 1, "Should parse one action with params"
    assert actions2[0].params.get("chunk_size") == 300, "Should have chunk_size param"
    assert actions2[0].params.get("extensions") == [".py"], "Should have extensions param"
    
    print("✓ memory.index_workspace action parses correctly")
    return True


def test_chunk_text_small():
    """Test chunking of small text (fits in one chunk)."""
    mem = Memory(kv_path=tempfile.mktemp(), vector_path=tempfile.mkdtemp())
    
    text = "This is a small text that fits in one chunk."
    chunks = mem._chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    assert len(chunks) == 1, "Small text should be one chunk"
    assert chunks[0]["text"] == text, "Text should be preserved"
    assert chunks[0]["chunk_index"] == 0, "Chunk index should be 0"
    assert chunks[0]["total_chunks"] == 1, "Total chunks should be 1"
    
    print("✓ _chunk_text handles small text correctly")
    return True


def test_chunk_text_large():
    """Test chunking of large text (splits into multiple chunks)."""
    mem = Memory(kv_path=tempfile.mktemp(), vector_path=tempfile.mkdtemp())
    
    # Create text larger than chunk_size
    lines = [f"Line {i}: This is some content to fill the chunk." for i in range(50)]
    text = "\n".join(lines)
    
    chunks = mem._chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    assert len(chunks) > 1, "Large text should be split into multiple chunks"
    
    # Verify chunk metadata
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i, f"Chunk {i} should have correct index"
        assert chunk["total_chunks"] == len(chunks), "All chunks should know total count"
        assert len(chunk["text"]) <= 600, "Chunk should be within size limit (with some tolerance)"
    
    print(f"✓ _chunk_text handles large text correctly ({len(chunks)} chunks)")
    return True


def test_index_workspace():
    """Test indexing a workspace with .py and .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        
        # Create test files
        (workspace / "main.py").write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""", encoding="utf-8")
        
        (workspace / "utils.py").write_text("""
def helper():
    return 42
""", encoding="utf-8")
        
        (workspace / "README.md").write_text("""
# Project Title

This is a sample project.

## Installation

Run `pip install -r requirements.txt`
""", encoding="utf-8")
        
        # Create memory instance
        vector_path = Path(tmpdir) / "vectors"
        mem = Memory(kv_path=str(Path(tmpdir) / "kv.json"), vector_path=str(vector_path))
        
        # Index workspace
        result = mem.index_workspace(workspace=str(workspace))
        
        assert result["files_processed"] == 3, f"Should process 3 files, got {result['files_processed']}"
        assert result["chunks_created"] > 0, "Should create at least one chunk"
        assert "error" not in result, "Should not have errors"
        
        # Verify vectors were stored
        stats = mem.stats()
        assert stats["vector"]["count"] == result["chunks_created"], "Vector count should match chunks created"
        
        print(f"✓ index_workspace indexed {result['files_processed']} files into {result['chunks_created']} chunks")
        return True


def test_index_workspace_custom_extensions():
    """Test indexing with custom file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        
        # Create test files
        (workspace / "code.py").write_text("x = 1\n", encoding="utf-8")
        (workspace / "data.txt").write_text("Some data\n", encoding="utf-8")
        (workspace / "notes.md").write_text("# Notes\n", encoding="utf-8")
        
        vector_path = Path(tmpdir) / "vectors"
        mem = Memory(kv_path=str(Path(tmpdir) / "kv.json"), vector_path=str(vector_path))
        
        # Index only .py files
        result = mem.index_workspace(workspace=str(workspace), extensions=[".py"])
        
        assert result["files_processed"] == 1, "Should only process .py file"
        
        print("✓ index_workspace respects custom extensions")
        return True


def test_index_workspace_empty():
    """Test indexing empty workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        
        vector_path = Path(tmpdir) / "vectors"
        mem = Memory(kv_path=str(Path(tmpdir) / "kv.json"), vector_path=str(vector_path))
        
        result = mem.index_workspace(workspace=str(workspace))
        
        assert result["files_processed"] == 0, "Should process 0 files"
        assert result["chunks_created"] == 0, "Should create 0 chunks"
        
        print("✓ index_workspace handles empty workspace")
        return True


if __name__ == "__main__":
    print("Testing memory.index_workspace Implementation...\n")
    
    tests = [
        test_action_type_exists,
        test_action_required_params,
        test_action_parsing,
        test_chunk_text_small,
        test_chunk_text_large,
        test_index_workspace,
        test_index_workspace_custom_extensions,
        test_index_workspace_empty,
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
