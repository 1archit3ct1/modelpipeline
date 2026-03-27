#!/usr/bin/env python
"""
Test script to verify the file.search (grep) functionality.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.environment.executor import Executor


def test_file_search_basic():
    """Test basic grep search functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exec = Executor(workspace=tmpdir)
        
        # Create test files
        Path(tmpdir, "test.py").write_text("""
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
""", encoding="utf-8")
        
        Path(tmpdir, "main.py").write_text("""
from test import hello

def main():
    hello()
""", encoding="utf-8")
        
        # Search for "def " pattern
        result = exec.file_search(pattern=r"def\s+\w+")
        
        assert "test.py" in result, "Should find matches in test.py"
        assert "main.py" in result, "Should find matches in main.py"
        assert "def hello" in result or "def goodbye" in result or "def main" in result, "Should find function definitions"
        
        print("✓ Basic file.search works")
        print(f"  Results:\n{result[:200]}")
        return True


def test_file_search_with_include():
    """Test grep search with file filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exec = Executor(workspace=tmpdir)
        
        # Create mixed files
        Path(tmpdir, "code.py").write_text("def func(): pass\n", encoding="utf-8")
        Path(tmpdir, "data.txt").write_text("def not_a_function\n", encoding="utf-8")
        
        # Search only in .py files
        result = exec.file_search(pattern="def", include_pattern="*.py")
        
        assert "code.py" in result, "Should find in .py file"
        assert "data.txt" not in result, "Should not find in .txt file when filtered"
        
        print("✓ file.search with include_pattern works")
        return True


def test_file_search_no_matches():
    """Test grep search with no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exec = Executor(workspace=tmpdir)
        
        Path(tmpdir, "test.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        
        result = exec.file_search(pattern="nonexistent_pattern_xyz123")
        
        assert "no matches" in result.lower(), "Should report no matches"
        
        print("✓ file.search reports no matches correctly")
        return True


def test_file_search_max_results():
    """Test grep search respects max_results limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exec = Executor(workspace=tmpdir)
        
        # Create file with many lines containing "test"
        content = "\n".join([f"line_{i}_test" for i in range(200)])
        Path(tmpdir, "big.py").write_text(content, encoding="utf-8")
        
        result = exec.file_search(pattern="test", max_results=10)
        
        # Count actual result lines (exclude the "limited" message)
        lines = [l for l in result.split("\n") if l.strip() and "limited" not in l.lower()]
        assert len(lines) <= 10, f"Should limit to 10 results, got {len(lines)}"
        assert "limited" in result.lower(), "Should indicate results were limited"
        
        print("✓ file.search respects max_results limit")
        return True


def test_file_search_action_type_exists():
    """Verify FILE_SEARCH action type is defined."""
    from src.agent.actions import ActionType
    
    assert hasattr(ActionType, "FILE_SEARCH"), "FILE_SEARCH action type should exist"
    assert ActionType.FILE_SEARCH.value == "file.search", "Should have correct value"
    
    print("✓ FILE_SEARCH action type is defined")
    return True


def test_file_search_required_params():
    """Verify FILE_SEARCH has required params defined."""
    from src.agent.actions import REQUIRED_PARAMS, ActionType
    
    assert ActionType.FILE_SEARCH in REQUIRED_PARAMS, "FILE_SEARCH should have required params"
    assert "pattern" in REQUIRED_PARAMS[ActionType.FILE_SEARCH], "pattern should be required"
    
    print("✓ FILE_SEARCH required params are defined")
    return True


if __name__ == "__main__":
    print("Testing file.search (grep) Implementation...\n")
    
    tests = [
        test_file_search_action_type_exists,
        test_file_search_required_params,
        test_file_search_basic,
        test_file_search_with_include,
        test_file_search_no_matches,
        test_file_search_max_results,
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
