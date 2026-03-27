#!/usr/bin/env python
"""
Test script to verify the TASK_CREATE execution handler.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.agent.actions import ActionType, parse_and_validate
from src.agent.tasks import TaskManager


def test_task_create_action_type_exists():
    """Verify TASK_CREATE action type is defined."""
    assert hasattr(ActionType, "TASK_CREATE"), "TASK_CREATE action type should exist"
    assert ActionType.TASK_CREATE.value == "task.create", "Should have correct value"
    print("✓ TASK_CREATE action type is defined")
    return True


def test_task_create_required_params():
    """Verify TASK_CREATE has required params defined."""
    from src.agent.actions import REQUIRED_PARAMS
    
    assert ActionType.TASK_CREATE in REQUIRED_PARAMS, "TASK_CREATE should have required params"
    assert "description" in REQUIRED_PARAMS[ActionType.TASK_CREATE], "description should be required"
    
    print("✓ TASK_CREATE required params are defined")
    return True


def test_task_create_parsing():
    """Verify task.create action parses correctly."""
    json_input = '{"action": "task.create", "description": "Implement feature X", "metadata": {"priority": "high"}}'
    actions = parse_and_validate(json_input)
    
    assert len(actions) == 1, "Should parse one action"
    assert actions[0].type == ActionType.TASK_CREATE, "Should be TASK_CREATE type"
    assert actions[0].params.get("description") == "Implement feature X", "Should have description"
    assert actions[0].valid, "Should be valid"
    
    print("✓ task.create action parses correctly")
    return True


def test_task_manager_create():
    """Verify TaskManager.create() works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(path=str(Path(tmpdir) / "tasks.jsonl"))
        
        task = tm.create("Test sub-task", {"parent": "main-task"})
        
        assert task is not None, "Should return a task"
        assert task.description == "Test sub-task", "Should have correct description"
        assert task.metadata.get("parent") == "main-task", "Should have metadata"
        assert task.id is not None, "Should have an ID"
        
        print("✓ TaskManager.create() works")
        return True


def test_task_create_in_runner():
    """Verify runner can execute task.create action."""
    from src.agent.runner import AgentRunner
    import os
    
    # Set up test environment
    test_data = tempfile.mkdtemp()
    test_workspace = tempfile.mkdtemp()
    os.environ["AGENT_WORKSPACE"] = test_workspace
    
    runner = AgentRunner(data_dir=test_data)
    
    # Create a mock action
    from src.agent.actions import Action
    
    action = Action(
        type=ActionType.TASK_CREATE,
        params={"description": "Sub-task: implement auth module", "metadata": {"step": 1}},
    )
    
    # Create and start a parent task
    current_task = runner.tasks.create("Parent task")
    runner.tasks.start(current_task.id)
    
    # Execute the action
    result = runner._execute(action, current_task)
    
    assert result is not None, "Should return created task"
    assert result.description == "Sub-task: implement auth module", "Should have correct description"
    assert result.metadata.get("step") == 1, "Should have metadata"
    
    # Verify task was added to task manager
    retrieved = runner.tasks.get(result.id)
    assert retrieved is not None, "Task should be retrievable"
    
    print("✓ Runner._execute() handles TASK_CREATE correctly")
    return True


if __name__ == "__main__":
    print("Testing TASK_CREATE Execution Handler...\n")
    
    tests = [
        test_task_create_action_type_exists,
        test_task_create_required_params,
        test_task_create_parsing,
        test_task_manager_create,
        test_task_create_in_runner,
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
