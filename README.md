# Agent Framework

Fully in-house, entirely offline autonomous agent. You own every byte.
This repository acts as a dual-value pipeline: it autonomously builds and ships real software stacks while continuously dumping premium transformer training data straight from the execution loop.

## Key Capabilities & Upgrades
* **Multi-Repo Control Surface:** The live Tauri GUI acts as a centralized command center. You can securely inject external customer repository paths, auto-trigger vector-memory ingestion, and instantly pivot the agent without ever restarting the engine.
* **15k Token Context Engine:** The `MAX_STATE_TOKENS` is optimized to 15,000, granting the agent a massive observation window to read files and debug.
* **Native Search & Recursion:** Equipped with cross-repository search utilities and `TASK_CREATE` scaffolding to recursively burn down complex goals.
* **Dynamic Dataset Metrics:** The GUI tracks live trajectory harvests alongside objective model parameter limits (7B, 15B, 30B). It visually monitors byte payload sizes and alerts you the second your fine-tuning dataset is structurally ready.
* **Real-Time Backend Trace:** An intercepted Python root logger actively pipes all raw execution traces and system limits safely into the Frontend DOM over JSON lines.

## How to Use Agent (Bootstrap Sequence)

All 14 core components are **already built and functional**. When the backend starts, it automatically runs a health check that verifies each component import and emits `component.verified` events — the GUI flips cards GREEN instantly.

**Single Bootstrap Task:**
Run one end-to-end verification task that exercises all components and generates your first training samples:

```bash
python tools/cli.py run "Verify the entire agent pipeline works end-to-end: 1) Read a file from the workspace, 2) Search for a function definition, 3) Store a test embedding to memory, 4) Retrieve it back, 5) Write a test file, 6) Run a shell command. Document each step so the reasoning is captured in training logs."
```

This single task:
- Validates all 14 components are GREEN and functional
- Generates 6-10 high-quality training samples
- Proves the observe→think→act loop works
- Bootstraps memory with initial vectors
- Confirms the bridge, executor, and model router all work together

### Standard Execution (Consistent Trigger)

Once the bootstrap task completes successfully, the agent is fully verified and all components are GREEN. From this point forward, use the CLI for any engineering task:

Because the ultimate goal of this repository is to **build a new, high-performing model from scratch**, trained specifically on complex software decisions, you can trigger the agent to work on deep architectural problems. Every observation, thought, and action will be permanently logged into the training trajectories.

Example deep engineering task:
```bash
python tools/cli.py run "Analyze the current vector indexing algorithm in store.py. Propose a mathematically superior routing logic for semantic similarity, and implement the changes. Document every technical abstraction thoroughly so the reasoning is captured perfectly in the training logs."
```

## Stack

| Layer | Tech |
|---|---|
| Agent loop, memory, training | Python |
| Embedding model | nomic-embed-text-v1.5 (local, GPU) |
| Vector index | hnswlib HNSW + numpy |
| Model backend | Ollama (local) / HuggingFace |
| GUI shell | Tauri (Rust) + plain HTML/JS |
| Bridge | stdin/stdout JSON lines |

## Quick Start

### 1. Install Python deps
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (recommended local backend)
Download from https://ollama.ai, then:
```bash
ollama pull mistral
```

### 3. Run via CLI
```bash
python tools/cli.py run "create a hello world Flask app"
```

### 4. Run via Tauri GUI
```bash
cd ui/src-tauri
cargo tauri dev
```
Set env vars first:
```
PYTHON_BIN=python
AGENT_SCRIPT=../../src/api.py
AGENT_WORKSPACE=../../workspace
```

## CLI Commands

```bash
# Run agent on a task
python tools/cli.py run "your task here" --model ollama:mistral --max-steps 100

# Show stats
python tools/cli.py stats

# Export training dataset
python tools/cli.py dataset --format alpaca

# Query memory
python tools/cli.py memory query "how to implement auth"

# Store to memory
python tools/cli.py memory store "completed auth module with JWT in 4 steps"
```

## Model Backends

```bash
# Local Ollama (default, recommended)
python tools/cli.py run "task" --model ollama:mistral

# Local HuggingFace
python tools/cli.py run "task" --model hf:mistralai/Mistral-7B-Instruct-v0.2
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `ollama:mistral` | Model backend |
| `AGENT_WORKSPACE` | `./workspace` | Sandboxed file ops root |
| `MODEL_CACHE` | `./data/model_cache` | HuggingFace cache dir |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `SHELL_TIMEOUT` | `30` | Shell command timeout (seconds) |

## Data Layout

```
data/
  memory/
    kv.json               ← short-term key-value store
    vectors/
      vectors.npy         ← float32 embedding matrix
      metadata.jsonl      ← per-vector metadata
      index.bin           ← HNSW index
      retrieval_log.jsonl ← Phase 2 training data
  trajectories/
    events.jsonl          ← all hook events (raw training data)
  datasets/               ← exported training datasets
  tasks.jsonl             ← task history
  state.json              ← last agent state
  graph.json              ← artifact graph
  model_cache/            ← HuggingFace model weights
workspace/                ← agent file operations sandbox
```

## GUI Roadmap

The Tauri GUI (`ui/frontend/index.html`) shows all system components as artifact cards.

- **RED** = component not yet built / not functional
- **GREEN** = component live and verified

As you build out each module, the corresponding card transitions from red to green automatically via events from the Python backend. If a card is green but broken in reality — **stop and fix before continuing.**

See `ROADMAP.md` for the full artifact registry and build order.

## Training Pipeline

Every agent run generates fine-tuning data automatically:

```bash
# After running the agent on some tasks:
python tools/cli.py dataset --format alpaca
# → data/datasets/alpaca_1234567890.json

# Use with any fine-tuning framework (Axolotl, LLaMA-Factory, etc.)
```

See `AGENTS.md` for full architecture documentation.
