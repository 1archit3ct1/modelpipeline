"""
actions.py — Action schema, parser, validator.
All agent outputs are parsed through here before execution.
"""

from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    # File operations
    FILE_CREATE   = "file.create"
    FILE_READ     = "file.read"
    FILE_EDIT     = "file.edit"
    FILE_DELETE   = "file.delete"
    FILE_LIST     = "file.list"
    FILE_SEARCH   = "file.search"   # grep search

    # Shell
    SHELL_RUN     = "shell.run"

    # Testing
    TEST_RUN      = "test.run"

    # Package management
    PIP_INSTALL   = "pip.install"
    PIP_FREEZE    = "pip.freeze"

    # Code quality
    LINT_RUN      = "lint.run"
    SECURITY_SCAN = "security.scan"

    # Git / Version control
    GIT_INIT      = "git.init"
    GIT_COMMIT    = "git.commit"
    GIT_STATUS    = "git.status"
    GIT_DIFF      = "git.diff"

    # Memory
    MEMORY_STORE         = "memory.store"
    MEMORY_QUERY         = "memory.query"
    MEMORY_INDEX_WORKSPACE = "memory.index_workspace"

    # Task control
    TASK_CREATE   = "task.create"
    TASK_COMPLETE = "task.complete"
    TASK_FAIL     = "task.fail"

    # Loop control
    LOOP_PAUSE    = "loop.pause"
    LOOP_STOP     = "loop.stop"

    # Output
    RESPOND       = "respond"
    THINK         = "think"   # internal reasoning, not executed


@dataclass
class Action:
    type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "params": self.params,
            "valid": self.valid,
            "error": self.error,
        }


# ── Parser ────────────────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON   = re.compile(r"\{[\s\S]*\}")


def parse(model_output: str) -> List[Action]:
    """
    Parse model output into a list of Actions.
    Supports:
    - ```json { "action": "file.create", ... } ```
    - bare JSON objects
    - multiple actions in one output
    """
    actions = []

    # try fenced blocks first
    blocks = _JSON_BLOCK.findall(model_output)
    if not blocks:
        blocks = _BARE_JSON.findall(model_output)

    for block in blocks:
        action = _parse_block(block)
        if action:
            actions.append(action)

    if not actions:
        # fall back: treat entire output as RESPOND
        actions.append(Action(
            type=ActionType.RESPOND,
            params={"text": model_output.strip()},
            raw=model_output,
        ))

    return actions


def _parse_block(block: str) -> Optional[Action]:
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed: {e}")
        return None

    raw_type = data.get("action") or data.get("type")
    if not raw_type:
        return None

    try:
        action_type = ActionType(raw_type)
    except ValueError:
        return Action(
            type=ActionType.RESPOND,
            params=data,
            raw=block,
            valid=False,
            error=f"Unknown action type: {raw_type}",
        )

    return Action(
        type=action_type,
        params={k: v for k, v in data.items() if k not in ("action", "type")},
        raw=block,
    )


# ── Validator ─────────────────────────────────────────────────────────────────

REQUIRED_PARAMS: Dict[ActionType, List[str]] = {
    ActionType.FILE_CREATE:   ["path", "content"],
    ActionType.FILE_READ:     ["path"],
    ActionType.FILE_EDIT:     ["path", "old", "new"],
    ActionType.FILE_DELETE:   ["path"],
    ActionType.FILE_SEARCH:   ["pattern"],
    ActionType.SHELL_RUN:     ["cmd"],
    ActionType.TEST_RUN:      [],  # all params optional
    ActionType.PIP_INSTALL:   ["packages"],
    ActionType.PIP_FREEZE:    [],
    ActionType.LINT_RUN:      [],
    ActionType.SECURITY_SCAN: [],
    ActionType.GIT_INIT:      [],
    ActionType.GIT_COMMIT:    ["message"],
    ActionType.GIT_STATUS:    [],
    ActionType.GIT_DIFF:      [],
    ActionType.MEMORY_STORE:  ["text"],
    ActionType.MEMORY_QUERY:  ["query"],
    ActionType.MEMORY_INDEX_WORKSPACE: [],  # all params optional
    ActionType.TASK_CREATE:   ["description"],
    ActionType.RESPOND:       ["text"],
}


def validate(action: Action) -> Action:
    required = REQUIRED_PARAMS.get(action.type, [])
    missing = [r for r in required if r not in action.params]
    if missing:
        action.valid = False
        action.error = f"Missing required params: {missing}"
    return action


def parse_and_validate(model_output: str) -> List[Action]:
    actions = parse(model_output)
    return [validate(a) for a in actions]
