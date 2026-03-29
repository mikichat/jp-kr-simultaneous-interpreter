@echo off
chcp 65001 > nul

echo.
echo  ================================================
echo   JP to KR Interpreter - Local LLM (Python)
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
    pip install --quiet faster-whisper
    echo [INFO] Packages installed.
)

echo [INFO] Checking Ollama server...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama...
    start "" ollama serve
    timeout /t 3 /nobreak > nul
)

echo [INFO] Starting translator...
echo.
python python\translator.py

call python\venv\Scripts\deactivate.bat
pause
