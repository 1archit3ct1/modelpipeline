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

- Max state tokens: **8,000** (leaves room for system prompt + response)
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
