@echo off
chcp 65001 > nul
setlocal

echo [INFO] JP-KR 동시통역 서버를 시작합니다...

taskkill /F /IM node.exe /T 2> nul

timeout /t 2 /nobreak > nul

if not exist "node_modules\" (
    echo [INFO] 의존성 패키지가 없습니다. 설치를 시작합니다...
    call npm install
)

echo [INFO] 개발 서버를 실행합니다...
echo [INFO] 브라우저에서 http://localhost:3000 에 접속하세요.
call npm run dev

endlocal
pause
