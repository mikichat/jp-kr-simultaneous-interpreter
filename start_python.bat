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
echo [INFO] Starting Llama.cpp Server mode...
echo [INFO] Starting llama-server...
start "" "C:\llama-cpp\llama-server" -m "C:\llama-cpp\Hy-MT2-1.8B-Q4_K_M.gguf" -c 2048 --port 8080
timeout /t 2 /nobreak > nul

echo [INFO] Starting translator...
echo.
python python\translator.py

call python\venv\Scripts\deactivate.bat
pause