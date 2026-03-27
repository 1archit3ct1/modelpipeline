# AGENTS.md — Agent Framework Specification

## What This Is

A fully in-house autonomous agent framework. No managed services. No black boxes.
Every component is owned, readable, and auditable. The entire stack is the training data.

---

## Architecture

```
User / Tauri GUI
      │
      │ JSON lines (stdin/stdout)
      ▼
  src/api.py  ←── entry point, spawned by Tauri as subprocess
      │
      ▼
AgentRunner (src/agent/runner.py)
  ├── ModelRouter      ← swap model without touching loop
  ├── Memory           ← two-tier: KV (exact) + VectorStore (semantic)
  ├── TaskManager      ← task lifecycle, step log
  ├── StateSerializer  ← token-budget-aware context builder
  ├── ArtifactGraph    ← tracks files/modules, red→green status
  ├── Executor         ← sandboxed file/shell operations
  └── Hooks            ← event bus → training data capture

VectorStore (src/vectorstore/)
  ├── embedder.py      ← nomic-embed-text-v1.5, local, GPU/CPU
  └── store.py         ← HNSW index + numpy array + jsonlines metadata

Training Pipeline (src/training/)
  └── pipeline.py      ← events.jsonl → (prompt, completion) pairs → JSONL/Alpaca/ShareGPT
```

---

## The Loop

```
OBSERVE  → build context from task + recent steps + memory snippets
THINK    → call model with context (≤8k tokens enforced)
PARSE    → extract action from model output (JSON block)
VALIDATE → check action schema and required params
EXECUTE  → run action via Executor or Memory
STORE    → emit training sample event
REPEAT   → until task.complete / task.fail / max_steps
```

---

## Tauri ↔ Python Bridge

Communication is JSON lines over stdin/stdout.

**Python → Tauri (stdout):**
```json
{"type": "step",    "step": 5, "max": 200}
{"type": "event",   "event": "action.executed", "payload": {...}}
{"type": "action",  "action": "file.create", "result": "created: auth.py"}
{"type": "respond", "text": "Task complete."}
{"type": "error",   "error": "model call failed"}
{"type": "stats",   "memory": {...}, "tasks": {...}}
{"type": "dataset", "path": "./data/datasets/...", "summary": {...}}
{"type": "ready",   "version": "0.1.0"}
```

**Tauri → Python (stdin):**
```json
{"cmd": "pause"}
{"cmd": "resume"}
{"cmd": "stop"}
{"cmd": "step"}
{"cmd": "task",       "description": "..."}
{"cmd": "run",        "task": "...", "model": "ollama:mistral"}
{"cmd": "set_model",  "model": "ollama:llama3"}
{"cmd": "memory_query","query": "..."}
{"cmd": "dataset",    "format": "alpaca"}
```

---

## Model Backends

Set via `AGENT_MODEL` env var or `--model` CLI flag.

| Value | Backend | Notes |
|---|---|---|
| `ollama:mistral` | Local Ollama | Default. Install from ollama.ai |
| `ollama:llama3`  | Local Ollama | Stronger reasoning |
| `ollama:codellama` | Local Ollama | Code-focused |
| `hf:mistral-7b`  | HuggingFace local | Requires model download |

---

## Action Schema

All model outputs are parsed as JSON action blocks:

```json
{"action": "file.create",   "path": "src/auth.py", "content": "..."}
{"action": "file.read",     "path": "src/auth.py"}
{"action": "file.edit",     "path": "src/auth.py", "old": "...", "new": "..."}
{"action": "file.list",     "path": "."}
{"action": "shell.run",     "cmd": "python -m pytest tests/"}
{"action": "memory.store",  "text": "Completed auth module in 3 steps"}
{"action": "memory.query",  "query": "how to implement JWT auth"}
{"action": "task.complete", "result": "all tests passing"}
{"action": "task.fail",     "error": "missing dependency"}
{"action": "respond",       "text": "Here is the result..."}
{"action": "think",         "text": "I need to check the schema first"}
```

---

## Context Budget

The state serializer enforces a hard token budget to keep model throughput high.

- Max state tokens: **15,000** (leaves room for system prompt + response)
- Priority order: current task > recent steps (last 6) > memory snippets (top 3) > KV vars
- Older steps are trimmed first. Memory snippets capped at 5.
- Never frontload the model. Context richness < throughput.

---

## Training Data Pipeline

Every agent run generates training data automatically.

1. **Capture**: `hooks.py` writes every event to `data/trajectories/events.jsonl`
2. **Extract**: `TrajectoryCapture.extract_samples()` pulls `training.sample` events
3. **Format**: `DatasetFormatter` outputs JSONL / Alpaca / ShareGPT
4. **Use**: Fine-tune any base model on your own agent's behavior

The retrieval log (`data/memory/vectors/retrieval_log.jsonl`) captures:
- What was queried
- What was returned
- What action followed

This becomes Phase 2 training data for the embedding model.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `ollama:mistral` | Model backend string |
| `AGENT_WORKSPACE` | `./workspace` | Sandboxed file operations root |
| `MODEL_CACHE` | `./data/model_cache` | HuggingFace model cache dir |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `SHELL_TIMEOUT` | `30` | Max seconds for shell commands |
| `PYTHON_BIN` | `python` | Python executable for Tauri to spawn |
| `AGENT_SCRIPT` | `./src/api.py` | Agent entry point for Tauri |

---

## Implementation Gaps & Required Tasks

> **CRITICAL DIRECTIVE FOR ALL AGENTS**
> Stop and ask for direction if you are unsure or unclear. Do not invent directions, paths, or wrapper scripts that are not explicitly authorized.

Currently, it is **impossible** for an autonomous agent spawning with no context to succeed under the strict 15k token limit due to the following structural gaps between the intended design and actual codebase:

### 1. Loop Trigger Gaps (`runner.py`)
- **Observation Truncation (The Blind Agent):** In `runner.py`, action outcomes are violently truncated: `"result": str(result)[:200]`. If an agent issues `file.read` or `shell.run`, it only gets back 200 characters in the loop's context. It cannot read code or debug effectively.
  - **Task Needed:** Create a specialized `current_observation` slot in `state.py` that pipes the raw, full result of the *immediately preceding* action (up to 4k tokens) directly into the agent's context, preserving the 200-char limits only for historical `recent_steps`.
- **No Native Search:**
  - **Task Needed:** Implement `file.search` (grep) in `executor.py` and `actions.py`. Without this, exploring a repository under a 15k limit by repeatedly running `file.list` and `file.read` is mathematically doomed.

### 2. Task Format Gaps (`tasks.py` & `runner.py`)
- **Missing Sub-task Execution Handler:** `ActionType.TASK_CREATE` is perfectly defined in `actions.py`, but completely absent in `runner.py`'s `_execute` block. If an agent tries to break down a larger goal into smaller task boundaries (crucial for a 15k context window), the action is swallowed and ignored.
  - **Task Needed:** Implement the target execution handler for `ActionType.TASK_CREATE` connecting directly to `self.tasks.create()`.

### 3. Memory Format Gaps (`memory.py` & Vector Store)
- **Dead-Start Indexing Problem:** The `VectorStore` initializes completely empty. Since `file.read` is truncated to 200 characters, the agent cannot manually read and chunk the workspace to store files into memory via `memory.store`. 
  - **Task Needed:** Implement a workspace-bootstrap hook on runner initialization, OR a dedicated `memory.index_workspace` action that automatically embeds all `.py`/`.md` files in the `AGENT_WORKSPACE`.

