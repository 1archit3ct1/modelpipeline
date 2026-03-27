# Automatically loads the Python backend environment variables
$env:PYTHON_BIN="python"
$env:AGENT_SCRIPT="../../src/api.py"
$env:AGENT_WORKSPACE="../../workspace"

Write-Host "Environment permanently loaded for this session." -ForegroundColor Cyan
Write-Host "Fetching pre-built Tauri binaries securely via NPX to bypass Rust compile bottlenecks..." -ForegroundColor Green

# Navigate to the rust/tauri backend and run the dev server
cd ui/src-tauri
npx @tauri-apps/cli@1 dev
