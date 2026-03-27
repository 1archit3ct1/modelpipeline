"""
graph.py — Artifact graph. Tracks files, modules, and their relationships.
Persists to JSON. Emits events on every mutation.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Node:
    id: str
    label: str
    type: str       # file | module | task | concept
    status: str     # red | green | unknown
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class ArtifactGraph:
    def __init__(self, path: str = "./data/graph.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for n in data.get("nodes", []):
                self._nodes[n["id"]] = Node(**n)
            for e in data.get("edges", []):
                self._edges.append(Edge(**e))

    def _save(self):
        from src.agent.hooks import emit, Event
        data = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "updated_at": time.time(),
        }
        self.path.write_text(json.dumps(data, indent=2))
        emit(Event.GRAPH_UPDATED, {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
        })

    def add_node(self, id: str, label: str, type: str = "file", status: str = "red", metadata: Optional[Dict] = None) -> Node:
        node = Node(id=id, label=label, type=type, status=status, metadata=metadata or {})
        self._nodes[id] = node
        self._save()
        return node

    def update_status(self, id: str, status: str):
        """Update node status — red (unbuilt) → green (live)."""
        if id in self._nodes:
            self._nodes[id].status = status
            self._nodes[id].updated_at = time.time()
            self._save()

    def add_edge(self, source: str, target: str, label: str = ""):
        self._edges.append(Edge(source=source, target=target, label=label))
        self._save()

    def get_node(self, id: str) -> Optional[Node]:
        return self._nodes.get(id)

    def to_dict(self) -> Dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def red_nodes(self) -> List[Node]:
        return [n for n in self._nodes.values() if n.status == "red"]

    def green_nodes(self) -> List[Node]:
        return [n for n in self._nodes.values() if n.status == "green"]
