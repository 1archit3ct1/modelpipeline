#!/usr/bin/env python
"""
Test script to verify the Observation Truncation fix.
Ensures that current_observation is properly passed to the agent context.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.agent.state import StateSerializer, MAX_OBSERVATION_TOKENS, CHARS_PER_TOKEN


def test_current_observation_included():
    """Test that current_observation appears in serialized state."""
    serializer = StateSerializer(path="./data/test_state.json")
    
    task = {"id": "test-1", "description": "Test task", "status": "running"}
    recent_steps = [
        {"action": "file.read", "summary": "read config.py"},
        {"action": "shell.run", "summary": "ran pytest"},
    ]
    
    # Simulate a large file read result (should be preserved up to 4k tokens)
    large_content = "# This is a test file\n" + "print('hello')\n" * 500
    current_observation = f"File content:\n{large_content}"
    
    context = serializer.serialize(
        task=task,
        recent_steps=recent_steps,
        current_observation=current_observation,
    )
    
    # Verify current observation is included
    assert "## Current Observation" in context, "Current observation section missing"
    assert "File content:" in context, "Observation content not in context"
    assert "print('hello')" in context, "Observation content truncated too aggressively"
    
    # Verify it respects the token budget
    max_chars = MAX_OBSERVATION_TOKENS * CHARS_PER_TOKEN
    obs_section_start = context.find("## Current Observation")
    obs_section_end = context.find("## Recent Steps")
    if obs_section_end == -1:
        obs_section_end = len(context)
    obs_section = context[obs_section_start:obs_section_end]
    
    print(f"✓ Current observation included in context")
    print(f"✓ Observation section length: {len(obs_section)} chars (max ~{max_chars})")
    print(f"✓ Full context length: {len(context)} chars")
    print("\nContext preview (first 500 chars):")
    print(context[:500])
    return True


def test_no_observation_when_none():
    """Test that no observation section appears when current_observation is None."""
    serializer = StateSerializer(path="./data/test_state.json")
    
    task = {"id": "test-1", "description": "Test task", "status": "running"}
    
    context = serializer.serialize(
        task=task,
        current_observation=None,
    )
    
    assert "## Current Observation" not in context, "Observation section should not appear when None"
    print("✓ No observation section when current_observation is None")
    return True


def test_observation_truncation():
    """Test that very large observations are truncated properly."""
    serializer = StateSerializer(path="./data/test_state.json")
    
    # Create content larger than MAX_OBSERVATION_TOKENS
    huge_content = "LINE\n" * 5000  # ~25k chars
    current_observation = f"Result:\n{huge_content}"
    
    context = serializer.serialize(
        task={"id": "test-1", "description": "Test", "status": "running"},
        current_observation=current_observation,
    )
    
    # Should contain truncation marker
    assert "...[truncated]..." in context, "Large observation should be truncated"
    print("✓ Large observations are properly truncated")
    return True


if __name__ == "__main__":
    print("Testing Observation Truncation Fix...\n")
    
    tests = [
        test_current_observation_included,
        test_no_observation_when_none,
        test_observation_truncation,
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
