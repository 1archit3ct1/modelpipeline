"""
executor.py — File system and shell executor.
Sandboxed to workspace directory. Shell commands run in subprocess.
"""

from __future__ import annotations
import os
import re
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

    def file_search(
        self,
        pattern: str,
        path: str = ".",
        include_pattern: Optional[str] = None,
        max_results: int = 100,
    ) -> str:
        """
        Grep search in files. Returns matching lines with file:line_number prefix.
        Pure Python implementation for cross-platform compatibility.

        Args:
            pattern: Regex pattern to search for
            path: Directory to search in (default: workspace root)
            include_pattern: Optional glob pattern to filter files (e.g., "*.py")
            max_results: Maximum number of matching lines to return
        """
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: directory not found: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"ERROR: invalid regex pattern: {e}"

        results = []
        files_searched = 0

        # Determine which files to search
        if include_pattern:
            files = p.rglob(include_pattern)
        else:
            files = p.rglob("*")

        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Skip binary files and large files
            try:
                if file_path.stat().st_size > 1_000_000:  # 1MB limit
                    continue
            except OSError:
                continue

            files_searched += 1
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        rel_path = file_path.relative_to(self.workspace)
                        results.append(f"{rel_path}:{line_num}:{line.rstrip()}")
                        
                        if len(results) >= max_results:
                            break
            except (OSError, IOError):
                continue  # Skip unreadable files
            
            if len(results) >= max_results:
                break

        if not results:
            return f"(no matches for pattern: {pattern})"

        output = "\n".join(results)
        if files_searched > 0 and len(results) >= max_results:
            output += f"\n...(results limited to {max_results} matches)"

        return output[:4000] if len(output) > 4000 else output

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

    # ── Testing ──────────────────────────────────────────────────────────────

    def test_run(self, path: str = ".", framework: str = "pytest") -> str:
        """Run tests using specified framework (pytest, unittest, etc.)."""
        if framework == "pytest":
            cmd = f"pytest {path} -v --tb=short"
        elif framework == "unittest":
            cmd = f"python -m unittest discover {path}"
        else:
            cmd = f"{framework} {path}"
        return self.shell_run(cmd)

    # ── Package Management ───────────────────────────────────────────────────

    def pip_install(self, packages: List[str], venv: Optional[str] = None) -> str:
        """Install Python packages using pip."""
        if not packages:
            return "ERROR: no packages specified"
        pip_cmd = "pip"
        if venv:
            venv_path = Path(venv)
            if os.name == "nt":  # Windows
                pip_cmd = str(venv_path / "Scripts" / "pip")
            else:
                pip_cmd = str(venv_path / "bin" / "pip")
        cmd = f"{pip_cmd} install {' '.join(packages)}"
        return self.shell_run(cmd)

    def pip_freeze(self, output: str = "requirements.txt") -> str:
        """Freeze installed packages to requirements file."""
        p = self._resolve(output)
        cmd = "pip freeze"
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=SHELL_TIMEOUT,
                cwd=str(self.workspace),
            )
            p.write_text(result.stdout, encoding="utf-8")
            return f"requirements frozen to {output}"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Code Quality ─────────────────────────────────────────────────────────

    def lint_run(self, path: str = ".", linter: str = "flake8") -> str:
        """Run linter on code."""
        if linter == "flake8":
            cmd = f"flake8 {path} --max-line-length=120"
        elif linter == "pylint":
            cmd = f"pylint {path}"
        elif linter == "ruff":
            cmd = f"ruff check {path}"
        else:
            cmd = f"{linter} {path}"
        return self.shell_run(cmd)

    def security_scan(self, path: str = ".") -> str:
        """Run security scan using bandit or safety."""
        # Try bandit first
        cmd = f"bandit -r {path} -ll"
        result = self.shell_run(cmd)
        if "No issues identified" in result or "issues identified: 0" in result:
            # Also check for known vulnerabilities
            cmd = "safety check"
            safety_result = self.shell_run(cmd)
            if "vulnerabilities found" not in safety_result.lower():
                return f"Security scan passed:\n{result}\n\nDependencies: {safety_result}"
        return f"Security scan:\n{result}"

    # ── Git / Version Control ────────────────────────────────────────────────

    def git_init(self) -> str:
        """Initialize git repository."""
        cmd = "git init"
        return self.shell_run(cmd)

    def git_commit(self, message: str, files: Optional[List[str]] = None) -> str:
        """Commit files to git."""
        cmds = []
        if files:
            for f in files:
                cmds.append(f"git add {f}")
        else:
            cmds.append("git add -A")
        cmds.append(f'git commit -m "{message}"')
        results = [self.shell_run(c) for c in cmds]
        return "\n".join(results)

    def git_status(self) -> str:
        """Show git status."""
        cmd = "git status"
        return self.shell_run(cmd)

    def git_diff(self, path: Optional[str] = None) -> str:
        """Show git diff."""
        if path:
            cmd = f"git diff {path}"
        else:
            cmd = "git diff"
        return self.shell_run(cmd)
