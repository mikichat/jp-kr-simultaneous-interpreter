@echo off
chcp 65001 > nul

echo.
echo  ================================================
echo   JP to KR Interpreter - Python Edition
echo  ================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed.
    pause
    exit /b 1
)

if not exist "python\venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv python\venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call python\venv\Scripts\activate.bat

python -c "import faster_whisper" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages...
    pip install --quiet -r python\requirements.txt
    pip install --quiet faster-whisper httpx
    echo [INFO] Packages installed.
)

echo.
echo  Select translation backend:
echo   1 - Ollama (local LLM)
echo   2 - Minimax API
echo.

set /p MODE="Enter 1 or 2: "

if "%MODE%"=="2" goto minimax

:ollama
echo [INFO] Starting with Ollama mode...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama server...
    start "" ollama serve
    timeout /t 3 /nobreak > nul
)
goto start_translator

:minimax
echo [INFO] Starting with Minimax API mode...
echo [INFO] Make sure MINIMAX_API_KEY environment variable is set.

:start_translator
echo [INFO] Starting translator...
echo.
python python\translator.py

call python\venv\Scripts\deactivate.bat
pause
