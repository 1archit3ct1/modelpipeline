# ROADMAP.md — GUI Artifact Conformance Guide

## How to Read This Document

**IMPORTANT:** All 14 components are **already built and functional**. 

- **RED** = component not yet verified (backend hasn't imported it yet)
- **GREEN** = component verified via import check on backend startup

When the Python backend starts, it runs `health_check()` which imports every component and emits `component.verified` events. The GUI receives these events and automatically flips cards from RED to GREEN within seconds of launch.

If a component shows GREEN but fails during actual use → **HALT and fix the bug.**

This document describes what each component does and how to verify it's working correctly.

---

## Artifact Registry

### Status Key
- 🔴 RED — not yet verified (will flip GREEN automatically on backend startup)
- 🟢 GREEN — verified and functional

---

## CORE PIPELINE

### 🔴 → 🟢 EMBEDDER
**GUI ID:** `art-embedder`
**File:** `src/vectorstore/embedder.py`
**Goes green when:**
- `nomic-embed-text-v1.5` weights downloaded to `MODEL_CACHE`
- `embed("test")` returns shape `(1, 768)` float32 array without error
- GPU detected and used if available (RTX 5090)
- `embed_query("test")` returns shape `(768,)` float32 array
**Metrics to verify:**
- Embed speed: target ≥2,000 embeddings/sec on GPU
- Model load time: <10s on first run (cached after)

---

### 🔴 → 🟢 VECTOR STORE
**GUI ID:** `art-vectorstore`
**File:** `src/vectorstore/store.py`
**Goes green when:**
- `VectorStore.store("text")` persists to `vectors.npy` + `metadata.jsonl`
- `VectorStore.retrieve("query", top_k=5)` returns correct results
- `hnswlib` index loads from `index.bin` on restart without rebuild
- Thread safety confirmed under concurrent store/retrieve
**Metrics to verify:**
- Retrieval latency: <5ms for 10k vectors (HNSW)
- Brute-force fallback active if hnswlib unavailable

---

### 🔴 → 🟢 MODEL ROUTER
**GUI ID:** `art-model-router`
**File:** `src/agent/model.py`
**Goes green when:**
- At least ONE backend responds to a test call without error
- `ollama:mistral` preferred (install Ollama first)
- Model swap via GUI dropdown triggers `set_model` cmd to Python
- `model.call(messages)` emits `model.call` and `model.response` events
**Metrics to verify:**
- First token latency logged per backend
- Error events emitted on failure (not silently swallowed)

---

## AGENT LOOP

### 🔴 → 🟢 RUNNER
**GUI ID:** `art-runner`
**File:** `src/agent/runner.py`
**Goes green when:**
- `runner.run("task")` completes at least one full loop iteration
- `loop.start` event emitted and received by GUI
- Pause/resume/stop/step controls functional via stdin bridge
- Step counter increments in GUI stat card
**Metrics to verify:**
- Steps per minute logged
- No silent failures — all errors surfaced to GUI event log

---

### 🔴 → 🟢 ACTIONS
**GUI ID:** `art-actions`
**File:** `src/agent/actions.py`
**Goes green when:**
- `parse_and_validate(model_output)` correctly extracts JSON action blocks
- All 10 action types parse without error on valid input
- Invalid actions return `valid=False` with error message (not crash)
- `action.executed` event emitted after successful execution
**Metrics to verify:**
- Parse success rate on real model output >95%

---

### 🔴 → 🟢 EXECUTOR
**GUI ID:** `art-executor`
**File:** `src/environment/executor.py`
**Goes green when:**
- `file_create`, `file_read`, `file_edit`, `file_delete`, `file_list` all functional
- Path traversal attack (`../../etc/passwd`) correctly blocked
- `shell_run` executes commands in workspace directory
- Shell timeout enforced at `SHELL_TIMEOUT` seconds
**Metrics to verify:**
- All file ops stay within `AGENT_WORKSPACE`
- Blocked commands return error string (not exception)

---

## MEMORY + STATE

### 🔴 → 🟢 MEMORY
**GUI ID:** `art-memory`
**File:** `src/agent/memory.py`
**Goes green when:**
- `memory.set(key, value)` persists to `kv.json`
- `memory.get(key)` returns correct value after process restart
- `memory.store(text)` stores to vector store and emits `memory.stored`
- `memory.retrieve(query)` returns scored results and emits `memory.retrieved`
- `memory.resume_context(task)` returns dict with kv_snapshot + memory_snippets
**Metrics to verify:**
- Resume context fits within 2k tokens
- KV + vector stats visible in GUI memory query panel

---

### 🔴 → 🟢 STATE SERIALIZER
**GUI ID:** `art-state`
**File:** `src/agent/state.py`
**Goes green when:**
- `state.serialize(task, recent_steps, memory_snippets)` returns string ≤8k tokens
- Long inputs are truncated cleanly (no mid-word cuts in critical fields)
- `state.save()` persists to `data/state.json` and emits `state.saved`
- `state.load()` restores on process restart with no-context resume
**Metrics to verify:**
- State string never exceeds `MAX_STATE_TOKENS * 4` characters
- Task description always present in output (highest priority)

---

### 🔴 → 🟢 ARTIFACT GRAPH
**GUI ID:** `art-graph`
**File:** `src/agent/graph.py`
**Goes green when:**
- Nodes persist to `data/graph.json` across restarts
- `update_status(id, "green")` correctly transitions node
- `graph.updated` event emitted on every mutation
- Graph canvas in GUI renders nodes and edges from live data
**Metrics to verify:**
- Node count and edge count visible in GUI graph overlay
- Red/green counts match GUI stat cards

---

## TRAINING PIPELINE

### 🔴 → 🟢 HOOKS / EVENTS
**GUI ID:** `art-hooks`
**File:** `src/agent/hooks.py`
**Goes green when:**
- All standard events (`Event.*`) emit without error during a run
- `data/trajectories/events.jsonl` grows during agent loop
- Events appear in GUI event log within 1 second of emission
- Hook callbacks execute without blocking the loop
**Metrics to verify:**
- Events per step logged (target: 4-8 events per step)
- Zero dropped events under normal operation

---

### 🔴 → 🟢 TRAJECTORY CAPTURE
**GUI ID:** `art-trajectory`
**File:** `src/training/pipeline.py` — `TrajectoryCapture`
**Goes green when:**
- `training.sample` events written to `events.jsonl` each loop step
- `extract_samples()` returns non-empty list after a completed run
- Each sample has `prompt`, `completion`, `result`, `step`, `timestamp`
- Sample count increments in GUI TRAIN SAMPLES stat card
**Metrics to verify:**
- Samples per run = steps completed (1:1 ratio)
- Avg prompt length: target 500-2000 chars

---

### 🔴 → 🟢 DATASET FORMATTER
**GUI ID:** `art-dataset`
**File:** `src/training/pipeline.py` — `DatasetFormatter`
**Goes green when:**
- EXPORT JSONL / ALPACA / SHAREGPT buttons trigger export
- Output files written to `data/datasets/`
- GUI event log shows path and sample count on export
- Files are valid JSON/JSONL parseable by standard tools
**Metrics to verify:**
- JSONL: one JSON object per line, no trailing commas
- Alpaca: `instruction` / `input` / `output` fields present
- ShareGPT: `conversations` array with `from` / `value` pairs

---

## TASKS

### 🔴 → 🟢 TASK MANAGER
**GUI ID:** `art-tasks`
**File:** `src/agent/tasks.py`
**Goes green when:**
- Tasks created via GUI task input appear in task queue panel
- Status transitions (pending→active→completed/failed) visible in GUI
- `data/tasks.jsonl` persists across restarts
- Task stats (pending/active/completed/failed counts) update in real time
**Metrics to verify:**
- Task completion rate tracked per session
- Failed tasks show error string in GUI

---

## BRIDGE + UI

### 🔴 → 🟢 STDIN/STDOUT BRIDGE
**GUI ID:** `art-bridge`
**File:** `src/api.py` + `ui/src-tauri/src/main.rs`
**Goes green when:**
- Tauri spawns Python subprocess on app start
- `{"type": "ready"}` message received by GUI within 2s of launch
- RUN button triggers Python loop via bridge
- PAUSE / STOP / STEP commands reach Python and take effect
- All GUI stat cards update from Python stdout events
**Metrics to verify:**
- Message round-trip latency: <100ms
- Zero message loss under normal operation
- Process cleanup on Tauri window close

---

### 🟢 GUI ROADMAP (this interface)
**GUI ID:** `art-gui`
**File:** `ui/frontend/index.html`
**Status:** LIVE — this file is built and functional.
**Remains green as long as:**
- All artifact cards render correctly
- Red/green transitions work on incoming events
- Graph canvas updates on status change
- Event log scrolls and displays all message types
- All control buttons send correct commands

---

## Bootstrap Verification

All components are already built. To verify everything works:

**Single Bootstrap Task:**
```bash
python tools/cli.py run "Verify the entire agent pipeline works end-to-end: read a file, search code, store to memory, retrieve from memory, write a file, run a shell command"
```

This task exercises all 14 components and generates your first training samples. The GUI will show all cards GREEN after the backend starts and this task completes.

---

## Halt Conditions

Stop production immediately if:
- A component shows GREEN but throws errors during actual use
- Token budget is exceeded (state serializer not enforcing limit)
- Training samples are not being captured during runs
- Path traversal is not blocked by executor
- Model calls are silently failing (no error event emitted)

Fix the violation before continuing with production tasks.
