# ============================================================
#  RAG Research Assistant - One-Click Startup Script
#  Usage:  .\start.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$env:OLLAMA_NO_CLOUD = "1"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$OLLAMA_EXE = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_STREAMLIT = Join-Path $PROJECT_DIR ".venv\Scripts\streamlit.exe"
$PORT = 8501
$OLLAMA_URL = "http://127.0.0.1:11434"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   RAG Research Assistant - Starting Local Environment  " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Check Ollama Binary ---
if (-not (Test-Path $OLLAMA_EXE)) {
    $found = Get-Command ollama -ErrorAction SilentlyContinue
    if ($found) { $OLLAMA_EXE = $found.Source }
    else {
        Write-Host "[ERROR] Ollama not found!" -ForegroundColor Red
        Write-Host "  Install: https://ollama.com/download/windows" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"; exit 1
    }
}
Write-Host "[OK] Ollama found: $OLLAMA_EXE" -ForegroundColor Green

# --- Step 2: Start Ollama if not running ---
$running = $false
try {
    $r = Invoke-WebRequest -Uri $OLLAMA_URL -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $running = $true }
} catch {}

if ($running) {
    Write-Host "[OK] Ollama server already running at $OLLAMA_URL" -ForegroundColor Green
} else {
    Write-Host "[..] Starting Ollama background server..." -ForegroundColor Yellow
    Start-Process $OLLAMA_EXE -ArgumentList "serve" -WindowStyle Minimized
    for ($i = 0; $i -lt 10; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri $OLLAMA_URL -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $running = $true; break }
        } catch {}
    }
    if ($running) { Write-Host "[OK] Ollama server started successfully" -ForegroundColor Green }
    else { Write-Host "[WARN] Ollama may still be starting..." -ForegroundColor Yellow }
}

# --- Step 3: Check embedding model ---
$list = & $OLLAMA_EXE list 2>&1 | Out-String
if ($list -match "nomic-embed-text") {
    Write-Host "[OK] Embedding model ready: nomic-embed-text" -ForegroundColor Green
} else {
    Write-Host "[..] Pulling nomic-embed-text ..." -ForegroundColor Yellow
    & $OLLAMA_EXE pull nomic-embed-text
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] Pulled: nomic-embed-text" -ForegroundColor Green }
    else { Write-Host "[WARN] Could not pull nomic-embed-text" -ForegroundColor Yellow }
}

# --- Step 4: Launch Streamlit and open browser ---
Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching Website: http://localhost:$PORT" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Opening your default web browser..." -ForegroundColor DarkGray
Write-Host "  Press Ctrl+C in this console to stop" -ForegroundColor DarkGray
Write-Host ""

# Free port if occupied by previous instance
$portConn = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
if ($portConn) {
    Write-Host "[..] Freeing port $PORT from previous process ($($portConn.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $portConn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

# Background job to open browser as soon as server starts
Start-Job -ScriptBlock {
    param($p)
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:$p"
} -ArgumentList $PORT | Out-Null

Set-Location $PROJECT_DIR
$VENV_PYTHON = Join-Path $PROJECT_DIR ".venv\Scripts\python.exe"
& $VENV_PYTHON -m streamlit run app.py --server.port $PORT --browser.gatherUsageStats false

