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

pip -q -r python\requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages...
    pip install -r python\requirements.txt
)

echo.
echo  Select translation backend:
echo   1 - Ollama (local LLM)
echo   2 - Minimax API
echo   3 - Llama.cpp Server
echo.

set /p MODE="Enter 1, 2, or 3: "

if "%MODE%"=="2" goto minimax
if "%MODE%"=="3" goto llamacpp

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
goto start_translator

:llamacpp
echo [INFO] Starting with Llama.cpp Server mode...
echo [INFO] Starting llama-server...
start "" "C:\llama-cpp\llama-server" -m "C:\llama-cpp\Hy-MT2-1.8B-Q4_K_M.gguf" -c 2048 --port 8080
timeout /t 2 /nobreak > nul
goto start_translator

:start_translator
echo [INFO] Starting translator...
echo.
python python\translator.py

call python\venv\Scripts\deactivate.bat
pause
