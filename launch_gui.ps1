# Agent Framework - Tauri GUI Launcher
# Sets up environment with absolute paths and launches Tauri

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set environment variables with absolute paths
$env:PYTHON_BIN = "python"
$env:AGENT_SCRIPT = "$PROJECT_ROOT\src\api.py"
$env:AGENT_WORKSPACE = "$PROJECT_ROOT\workspace"
$env:AGENT_MODEL = "ollama:mistral"

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  AGENT FRAMEWORK - TAURI GUI" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Gray
Write-Host "Python Script: $env:AGENT_SCRIPT" -ForegroundColor Gray
Write-Host "Workspace: $env:AGENT_WORKSPACE" -ForegroundColor Gray
Write-Host "Model: $env:AGENT_MODEL" -ForegroundColor Gray
Write-Host ""
Write-Host "Launching Tauri dev server..." -ForegroundColor Green
Write-Host ""

# Navigate to Tauri backend and run
cd "$PROJECT_ROOT\ui\src-tauri"
npx @tauri-apps/cli@1 dev
