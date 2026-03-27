# Agent Framework

Fully in-house autonomous agent. You own every byte.
Custom transformer training data pipeline built into the loop itself.

## How to Use Agent (Bootstrap Sequence)

Because the agent framework requires the agent to fix its own initialization blind spots natively, we use a sequenced approach. Run the following prompts via the CLI in order:

**Prompt 1: Un-blinding the Agent**
This prompt orders the agent to fix its own state truncation limit, allowing it to "see" its environment fully.
```bash
python tools/cli.py run "Fix the Observation Truncation blind spot in runner.py and state.py by passing the raw execution result into context"
```

**Prompt 2: Building Core Tools**
Once the agent's engine restarts and it can see 4k context blocks, instruct it to build the cross-repository native search. 
```bash
python tools/cli.py run "Implement a file.search grep tool in executor.py and actions.py allowing precise code retrieval"
```

**Prompt 3: Enabling Sub-Tasking**
The agent needs to be able to spawn sub-tasks to protect its 15k token limit. Instruct it to wire up the task creation logic.
```bash
python tools/cli.py run "Implement the execution handler for ActionType.TASK_CREATE in runner.py's _execute block so I can break down large tasks into smaller sub-tasks"
```

**Prompt 4: Memory Bootstrapping**
Finally, fix the dead-start memory problem so the agent can autonomously populate its own vector database from the repository contents.
```bash
python tools/cli.py run "Implement a memory.index_workspace action in actions.py and memory.py that automatically chunks and embeds all .py and .md files in the workspace into the VectorStore"
```

### Standard Execution (Consistent Trigger)

Once the 4 bootstrapping prompts are successfully executed, the agent's core architecture is fully patched and un-blinded. From this point forward, the CLI command becomes **consistent**. 

Because the ultimate goal of this repository is to **build a new, high-performing model from scratch**, trained specifically on complex software decisions, you can now trigger the agent to work on deep architectural problems. Every observation, thought, and action will be permanently securely logged into the training trajectories.

Here is an example of the ultimate, consistent prompt to run:
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
