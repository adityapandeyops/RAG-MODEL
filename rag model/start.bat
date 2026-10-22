@echo off
setlocal enabledelayedexpansion
title RAG Research Assistant
echo.
echo ========================================================
echo   RAG Research Assistant - Starting Local Environment
echo ========================================================
echo.

set OLLAMA_NO_CLOUD=1
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
set PORT=8501
set OLLAMA_URL=http://127.0.0.1:11434

REM --- Check Ollama Binary ---
if not exist "%OLLAMA_EXE%" (
    for /f "tokens=*" %%i in ('where ollama 2^>nul') do set OLLAMA_EXE=%%i
)
if not exist "%OLLAMA_EXE%" (
    echo [ERROR] Ollama not found! Install from https://ollama.com/download/windows
    pause
    exit /b 1
)
echo [OK] Ollama found: %OLLAMA_EXE%

REM --- Start Ollama server in background if not running ---
curl -s %OLLAMA_URL% >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] Starting Ollama background server...
    start "" /min "%OLLAMA_EXE%" serve
    echo      Waiting for Ollama to become ready...
    for /l %%i in (1,1,10) do (
        timeout /t 1 /nobreak >nul
        curl -s %OLLAMA_URL% >nul 2>&1
        if !errorlevel! equ 0 goto :ollama_ready
    )
)
:ollama_ready
curl -s %OLLAMA_URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama server is running at %OLLAMA_URL%
) else (
    echo [WARN] Ollama server is taking longer to start. Continuing...
)

REM --- Verify embedding model ---
"%OLLAMA_EXE%" list 2>nul | findstr /i "nomic-embed-text" >nul
if %errorlevel% neq 0 (
    echo [..] Pulling nomic-embed-text embedding model
    "%OLLAMA_EXE%" pull nomic-embed-text
)
echo [OK] nomic-embed-text embedding model ready

echo.
echo ========================================================
echo   Launching Website at: http://localhost:%PORT%
echo ========================================================
echo   Opening your default web browser automatically...
echo   Press Ctrl+C in this window when you want to stop.
echo.

REM --- Ensure port is free from any previous orphan process ---
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo [..] Freeing port %PORT% from previous instance...
    taskkill /f /pid %%a >nul 2>&1
)

REM Automatically open default browser after brief delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:%PORT%"

cd /d "%~dp0"
"%VENV_PYTHON%" -m streamlit run app.py --server.port %PORT% --browser.gatherUsageStats false


