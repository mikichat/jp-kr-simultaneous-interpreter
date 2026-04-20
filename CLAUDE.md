# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

일본어/영어/중국어 → 한국어 동시통역 프로젝트. 두 가지 구현체가 존재:

- **Web UI** (`src/App.tsx`): React + Gemini API 기반 브라우저 번역기
- **Python CLI** (`python/translator.py`): Terminal 기반 로컬 Whisper + Ollama/Minimax 번역기

## 개발 명령어

### Frontend (React/Vite)

```sh
npm run dev      # 개발 서버 실행 (포트 3000)
npm run build    # 프로덕션 빌드
npm run lint     # TypeScript 타입 검사
npm run preview  # 빌드 결과 미리보기
```

### Python CLI

```sh
# 가상환경 활성화 후
cd python
source venv/bin/activate  # Windows: venv\Scripts\activate
python translator.py

# 의존성 설치
pip install -r python/requirements.txt
```

## 환경변수

| 변수 | 설명 | 위치 |
|------|------|------|
| `GEMINI_API_KEY` | Gemini API 키 (Web UI용) | `.env` 또는 `aistudio.google.com`에서 발급 |
| `MINIMAX_API_KEY` | Minimax API 키 (Python CLI용, 선택) | 환경변수 또는 `translator.py` 내 설정 |

## 아키텍처

### Web UI 파이프라인

```text
마이크/스테레오믹스 → MediaRecorder (4초 청크) → Gemini 2.0 Flash Lite → (일본어 전사 + 한국어 번역)
```

### Python CLI 파이프라인 (3단계)

```text
오디오 입력 → [Audio Collector] → stt_queue
                                        ↓
                              [STT Worker: faster-whisper] → translate_queue
                                                          ↓
                                              [Translate Worker: Ollama/Minimax] → 화면 출력
```

## 주요 설정값

### Python CLI (`python/translator.py`)

- `SAMPLE_RATE = 16000` (Whisper 권장)
- `CHUNK_SEC = 1` (오디오 청크 크기)
- `WHISPER_MODEL = "small"` (tiny/base/small/medium)
- `OLLAMA_HOST = "http://localhost:11434"`
- 소스 언어: `ja` (일본어), `en` (영어), `zh` (중국어), `auto` (자동감지)

### 런타임 변경 (Python CLI)

- `D` 키: 오디오 장치 변경
- `M` 키: Ollama 모델 변경
- `Ctrl+C`: 종료

## 디렉토리 구조

```text
├── src/                 # React Frontend (Vite)
│   ├── App.tsx         # 메인 번역기 UI
│   ├── main.tsx       # 진입점
│   └── index.css      # Tailwind CSS
├── python/
│   ├── translator.py  # Python CLI 동시통역기
│   └── requirements.txt
├── vite.config.ts
├── package.json
└── index.html
```

## 참고사항

- **PC 사운드 캡처**: Windows에서 "스테레오 믹스" 또는 "WASAPI Loopback" 장치 필요
- **Python CLI 실행 전**: Ollama (`ollama serve`) 실행 중인지 확인
- **Web UI 실행 전**: `.env` 파일에 `GEMINI_API_KEY` 설정

## 사용 언어

반드시 한국어로 응답

## Git 워크플로우

1. 파일 수정 전: `git pull` 실행
2. 파일 수정 후: `git push` 실행
3. 커밋 메시지: 반드시 한국어로 작성
