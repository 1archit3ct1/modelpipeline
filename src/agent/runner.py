"""
runner.py — The main agent loop.
Run → Observe → Think → Act → Store → Repeat
Pause/Stop/Step controlled via stdin JSON commands from Tauri.
"""

from __future__ import annotations
import json
import sys
import time
import select
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous agent. You operate a loop:
1. Read current state and task
2. Decide on ONE action
3. Output ONLY a JSON action block

Available actions:
- file.create:   {"action": "file.create", "path": "...", "content": "..."}
- file.read:     {"action": "file.read", "path": "..."}
- file.edit:     {"action": "file.edit", "path": "...", "old": "...", "new": "..."}
- file.list:     {"action": "file.list", "path": "..."}
- file.search:   {"action": "file.search", "pattern": "...", "path": ".", "include_pattern": "*.py"}
- shell.run:     {"action": "shell.run", "cmd": "..."}
- memory.store:  {"action": "memory.store", "text": "..."}
- memory.query:  {"action": "memory.query", "query": "..."}
- task.complete: {"action": "task.complete", "result": "..."}
- task.fail:     {"action": "task.fail", "error": "..."}
- respond:       {"action": "respond", "text": "..."}
- think:         {"action": "think", "text": "..."}

Be concise. One action per response. No preamble."""


class AgentRunner:
    def __init__(
        self,
        model_str: Optional[str] = None,
        data_dir: str = "./data",
    ):
        from src.agent.hooks import init as init_hooks, emit, Event
        from src.agent.model import ModelRouter
        from src.agent.memory import Memory
        from src.agent.tasks import TaskManager
        from src.agent.state import StateSerializer
        from src.agent.graph import ArtifactGraph
        from src.environment.executor import Executor

        init_hooks(log_dir=f"{data_dir}/trajectories")

        self.model = ModelRouter(model_str)
        self.memory = Memory(
            kv_path=f"{data_dir}/memory/kv.json",
            vector_path=f"{data_dir}/memory/vectors",
        )
        self.tasks = TaskManager(path=f"{data_dir}/tasks.jsonl")
        self.state = StateSerializer(path=f"{data_dir}/state.json")
        self.graph = ArtifactGraph(path=f"{data_dir}/graph.json")
        self.executor = Executor()

        self._paused = False
        self._stopped = False
        self._step_mode = False
        self._step_event = threading.Event()

        # start stdin listener thread
        self._stdin_thread = threading.Thread(target=self._listen_stdin, daemon=True)
        self._stdin_thread.start()

        self._emit = emit
        self._Event = Event

    # ── Stdin bridge (Tauri → Python) ─────────────────────────────────────────

    def _listen_stdin(self):
        """
        Read JSON commands from stdin. Tauri sends these to control the loop.
        Commands: {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "stop"} | {"cmd": "step"}
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                cmd = msg.get("cmd", "")
                if cmd == "pause":
                    self._paused = True
                    self._emit(self._Event.LOOP_PAUSE, {})
                elif cmd == "resume":
                    self._paused = False
                    self._emit(self._Event.LOOP_RESUME, {})
                elif cmd == "stop":
                    self._stopped = True
                    self._emit(self._Event.LOOP_STOP, {"reason": "user"})
                elif cmd == "step":
                    self._step_mode = True
                    self._step_event.set()
                elif cmd == "task":
                    self.tasks.create(msg.get("description", "New task"))
            except json.JSONDecodeError:
                pass

    def _bridge(self, type_: str, data: Dict):
        """Send structured JSON line to Tauri via stdout."""
        print(json.dumps({"type": type_, **data}), flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, task_description: str, max_steps: int = 200):
        task = self.tasks.create(task_description)
        self.tasks.start(task.id)
        self._emit(self._Event.LOOP_START, {"task_id": task.id, "description": task_description})
        self._bridge("status", {"state": "running", "task_id": task.id})

        step = 0
        recent_steps = []
        last_action_result = None  # raw result of immediately preceding action

        while step < max_steps and not self._stopped:
            # pause check
            while self._paused and not self._stopped:
                time.sleep(0.2)

            if self._stopped:
                break

            # step mode — wait for explicit step signal
            if self._step_mode:
                self._step_event.wait()
                self._step_event.clear()

            step += 1
            self._bridge("step", {"step": step, "max": max_steps})

            # ── build context ──────────────────────────────────────────────
            resume = self.memory.resume_context(task_description)
            context_str = self.state.serialize(
                task=task.to_dict(),
                recent_steps=recent_steps[-6:],
                memory_snippets=resume["memory_snippets"],
                agent_vars=resume["kv_snapshot"],
                current_observation=last_action_result,  # raw result from last action
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context_str},
            ]

            # ── model call ─────────────────────────────────────────────────
            try:
                raw_output = self.model.call(messages, max_tokens=512, temperature=0.2)
            except Exception as e:
                logger.error(f"Model call failed: {e}")
                self._bridge("error", {"error": str(e)})
                time.sleep(2)
                continue

            # ── parse + validate actions ───────────────────────────────────
            from src.agent.actions import parse_and_validate
            actions = parse_and_validate(raw_output)

            for action in actions:
                if not action.valid:
                    self._bridge("action_invalid", action.to_dict())
                    continue

                self._emit(self._Event.ACTION_PROPOSED, action.to_dict())

                # ── execute ────────────────────────────────────────────────
                result = self._execute(action, task)

                # Store raw result for next iteration's current_observation
                last_action_result = str(result)

                step_record = {
                    "step": step,
                    "action": action.type.value,
                    "params": action.params,
                    "result": str(result)[:200],  # truncated for history
                    "summary": f"{action.type.value}: {str(result)[:100]}",
                }
                recent_steps.append(step_record)
                self.tasks.add_step(task.id, step_record)

                # capture training sample
                self._emit(self._Event.TRAINING_SAMPLE, {
                    "step": step,
                    "context": context_str[:500],
                    "action": action.to_dict(),
                    "result": str(result)[:200],
                })

                # check for terminal actions
                if action.type.value in ("task.complete", "task.fail", "loop.stop"):
                    self._stopped = True
                    break

        # ── wrap up ────────────────────────────────────────────────────────
        if not self._stopped:
            self.tasks.fail(task.id, "max steps reached")
        self._bridge("status", {"state": "stopped", "steps": step})
        self._emit(self._Event.LOOP_STOP, {"steps": step})

    # ── Action executor ───────────────────────────────────────────────────────

    def _execute(self, action, task) -> Any:
        from src.agent.actions import ActionType
        t = action.type
        p = action.params

        try:
            if t == ActionType.FILE_CREATE:
                result = self.executor.file_create(p["path"], p["content"])
            elif t == ActionType.FILE_READ:
                result = self.executor.file_read(p["path"])
            elif t == ActionType.FILE_EDIT:
                result = self.executor.file_edit(p["path"], p["old"], p["new"])
            elif t == ActionType.FILE_DELETE:
                result = self.executor.file_delete(p["path"])
            elif t == ActionType.FILE_LIST:
                result = self.executor.file_list(p.get("path", "."))
            elif t == ActionType.FILE_SEARCH:
                result = self.executor.file_search(
                    pattern=p["pattern"],
                    path=p.get("path", "."),
                    include_pattern=p.get("include_pattern"),
                    max_results=p.get("max_results", 100),
                )
            elif t == ActionType.SHELL_RUN:
                result = self.executor.shell_run(p["cmd"])
            elif t == ActionType.MEMORY_STORE:
                result = self.memory.store(p["text"], p.get("metadata"))
            elif t == ActionType.MEMORY_QUERY:
                result = self.memory.retrieve(p["query"], top_k=p.get("top_k", 5))
            elif t == ActionType.TASK_COMPLETE:
                result = self.tasks.complete(task.id, p.get("result", ""))
                self._stopped = True
            elif t == ActionType.TASK_FAIL:
                result = self.tasks.fail(task.id, p.get("error", ""))
                self._stopped = True
            elif t == ActionType.RESPOND:
                result = p.get("text", "")
                self._bridge("respond", {"text": result})
            elif t == ActionType.THINK:
                result = p.get("text", "")
                self._bridge("think", {"text": result})
            elif t == ActionType.LOOP_STOP:
                self._stopped = True
                result = "loop stopped"
            else:
                result = f"unhandled action: {t}"

            self._emit(self._Event.ACTION_EXECUTED, {
                "action": t.value,
                "result": str(result)[:200],
            })
            self._bridge("action", {"action": t.value, "result": str(result)[:200]})
            return result

        except Exception as e:
            self._emit(self._Event.ACTION_FAILED, {"action": t.value, "error": str(e)})
            self._bridge("error", {"action": t.value, "error": str(e)})
            return f"ERROR: {e}"
