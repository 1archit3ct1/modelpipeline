"""
executor.py — File system and shell executor.
Sandboxed to workspace directory. Shell commands run in subprocess.
"""

from __future__ import annotations
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

WORKSPACE = os.environ.get("AGENT_WORKSPACE", "./workspace")
SHELL_TIMEOUT = int(os.environ.get("SHELL_TIMEOUT", "30"))
BLOCKED_CMDS = {"rm -rf /", "mkfs", "dd if=/dev/zero", ":(){ :|:& };:"}


class Executor:
    def __init__(self, workspace: str = WORKSPACE):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        logger.info(f"Executor workspace: {self.workspace}")

    def _resolve(self, path: str) -> Path:
        """Resolve path inside workspace. Prevents path traversal."""
        resolved = (self.workspace / path).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise PermissionError(f"Path traversal blocked: {path}")
        return resolved

    def file_create(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"created: {path} ({len(content)} bytes)"

    def file_read(self, path: str, max_chars: int = 16000) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: file not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return content

    def file_edit(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: file not found: {path}"
        content = p.read_text(encoding="utf-8")
        if old not in content:
            return f"ERROR: old string not found in {path}"
        p.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"edited: {path}"

    def file_delete(self, path: str) -> str:
        p = self._resolve(path)
        if p.exists():
            p.unlink()
            return f"deleted: {path}"
        return f"not found: {path}"

    def file_list(self, path: str = ".") -> List[str]:
        p = self._resolve(path)
        if not p.exists():
            return []
        return [str(f.relative_to(self.workspace)) for f in p.rglob("*") if f.is_file()][:200]

    def shell_run(self, cmd: str) -> str:
        # basic safety check
        for blocked in BLOCKED_CMDS:
            if blocked in cmd:
                return f"BLOCKED: dangerous command pattern"

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=SHELL_TIMEOUT,
                cwd=str(self.workspace),
            )
            out = (result.stdout + result.stderr).strip()
            return out[:4000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {SHELL_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: {e}"
