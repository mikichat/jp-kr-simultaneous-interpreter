@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

echo.
echo  ================================================
echo   JP to KR Interpreter (Python)
echo   Ollama 또는 Minimax API 선택 가능
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

:: 번역 모드 선택
set TRANSLATION_MODE=
if "%1"=="" (
    echo.
    echo  번역 백엔드를 선택하세요:
    echo   [1] Ollama (로컬 LLM - 무료, GPU 권장)
    echo   [2] Minimax API (클라우드 - API 키 필요)
    echo.
    set /p MODE_CHOICE="번호를 입력하세요 (1 또는 2): "
    if "!MODE_CHOICE!"=="2" set TRANSLATION_MODE=minimax
) else (
    if "%1"=="minimax" set TRANSLATION_MODE=minimax
)

if "!TRANSLATION_MODE!"=="minimax" (
    echo [INFO] Minimax API 모드로 실행합니다.
    if "%MINIMAX_API_KEY%"=="" (
        echo.
        echo  ⚠️  MINIMAX_API_KEY 환경변수가 설정되지 않았습니다.
        echo  translator.py에서 API 키를 입력하거나 환경변수를 설정하세요.
        echo.
        set /p API_KEY="Minimax API 키를 입력하세요: "
        setx MINIMAX_API_KEY "!API_KEY!" >nul
        set MINIMAX_API_KEY=!API_KEY!
        echo [INFO] API 키가 환경변수로 설정되었습니다.
    )
) else (
    echo [INFO] Ollama 모드로 실행합니다.
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Ollama 서버를 시작합니다...
        start "" ollama serve
        timeout /t 3 /nobreak > nul
    )
)

echo [INFO] 번역기를 시작합니다...
echo.
python python\translator.py

call python\venv\Scripts\deactivate.bat
pause
