"""
tasks.py — Task list and status lifecycle.
States: pending → active → completed | failed | cancelled
"""

from __future__ import annotations
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**d)


class TaskManager:
    def __init__(self, path: str = "./data/tasks.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, Task] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                for line in f:
                    if line.strip():
                        t = Task.from_dict(json.loads(line))
                        self._tasks[t.id] = t
            logger.info(f"Loaded {len(self._tasks)} tasks")

    def _save(self):
        with open(self.path, "w") as f:
            for t in self._tasks.values():
                f.write(json.dumps(t.to_dict()) + "\n")

    def create(self, description: str, metadata: Optional[Dict] = None) -> Task:
        from src.agent.hooks import emit, Event
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        self._save()
        emit(Event.TASK_CREATED, task.to_dict())
        return task

    def start(self, task_id: str) -> Task:
        from src.agent.hooks import emit, Event
        task = self._tasks[task_id]
        task.status = TaskStatus.ACTIVE
        task.started_at = time.time()
        self._save()
        emit(Event.TASK_STARTED, task.to_dict())
        return task

    def complete(self, task_id: str, result: str = "") -> Task:
        from src.agent.hooks import emit, Event
        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        self._save()
        emit(Event.TASK_COMPLETED, task.to_dict())
        return task

    def fail(self, task_id: str, error: str = "") -> Task:
        from src.agent.hooks import emit, Event
        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.error = error
        self._save()
        emit(Event.TASK_FAILED, task.to_dict())
        return task

    def add_step(self, task_id: str, step: Dict):
        task = self._tasks[task_id]
        task.steps.append({**step, "timestamp": time.time()})
        self._save()

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list(self, status: Optional[TaskStatus] = None) -> List[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at)

    def stats(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in TaskStatus}
        for t in self._tasks.values():
            counts[t.status.value] += 1
        return counts
