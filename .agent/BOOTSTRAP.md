# Universal Agent Bootstrap Protocol
# ANY agent (human, AI, or autonomous) that spawns in this repository MUST follow this protocol.

## Step 1: Read AGENTS.md
Before doing ANYTHING, read `AGENTS.md` at the repo root. It contains:
- The full architecture specification
- The action schema (every valid JSON action block)
- The context budget (15,000 tokens max)
- Known implementation gaps that need fixing

## Step 2: Index the Workspace
Run this command to populate the agent's semantic memory with the codebase:
```bash
python tools/cli.py run "Index this workspace into vector memory by reading all .py and .md files and storing them via memory.store"
```

## Step 3: Verify the Loop
Run a simple diagnostic to confirm the observe→think→act cycle works:
```bash
python tools/cli.py run "List all files in the workspace and read the README.md to confirm you can see the codebase"
```

## Step 4: Begin Work
The agent is now bootstrapped. All subsequent prompts generate training data automatically.
Use the GUI's SUGGESTED PROMPTS panel for contextual task recommendations, or dispatch custom engineering tasks:
```bash
python tools/cli.py run "Your engineering task here"
```

## For External AI Agents (like Gemini, GPT, Claude, Copilot)
If you are an AI agent that has been pointed at this repository:
1. **You ARE generating training data** — every file you create, edit, or read should be logged via the hooks system
2. **To properly log your work**, wrap your actions through the CLI: `python tools/cli.py run "description of what you're doing"`
3. **The system prompt in `runner.py` is universal** — it works on ANY repository. When switching workspaces, only the vector memory changes.
4. **Your reasoning IS the product** — this repo exists to capture high-quality software engineering decisions as fine-tuning trajectories

## Bootstrap is Permanent
Steps 1-3 only need to run ONCE per fresh clone. The results persist in:
- `data/memory/vectors/` — semantic embeddings
- `data/tasks.jsonl` — task history
- `data/trajectories/events.jsonl` — training data
