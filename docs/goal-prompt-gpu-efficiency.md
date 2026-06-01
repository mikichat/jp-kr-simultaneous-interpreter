# Goal Prompt: GPU 자원 낭비 방지 및 효율적인 로컬 LLM 번역

## 🎯 목표

로컬 LLM 번역 파이프라인에서 GPU 자원을 낭비하지 않고 효율적으로 동작하도록 최적화한다.

---

## 📋 체크리스트 (완료 현황: 2026-06-01)

### 1. Llama.cpp Server 설정 최적화
- [x] `LLAMACPP_HOST`, `LLAMACPP_MODEL` 환경변수로 분리 (하드코딩 제거)
- [x] `temperature` 0.3 이하로 고정 (`LLM_TEMPERATURE` 환경변수, 기본값 0.3)
- [x] `num_ctx` 2048로 설정 (`LLM_NUM_CTX` 환경변수)
- [x] `LLM_GPU_LAYERS` 환경변수로 GPU 사용량 명시적 제한
- [n/a] `numa` - Llama.cpp Server 실행 시 `--numa` 플래그로 설정 (클라이언트에서 제어 불가)

### 2. 번역 워커 병렬 처리 제어
- [x] `TRANSLATE_WORKERS` 기본값 `1` 유지
- [x] 세마포어 구현 (`translation_semaphore`, 동시 번역 1개로 제한)
- [x] `translate_queue.qsize() > 15` 시 요청 거부 (메모리 보호)

### 3. STT (Whisper) GPU 사용 안 함
- [x] `device="cpu"` 고정
- [x] `compute_type="int8"` 고정
- [x] 배치 크기 `1` 고정

### 4. 메모리 관리
- [x] `MAX_AUDIO_BUFFER_SEC = 30` 오디오 버퍼 크기 명시적 제한
- [x] `MAX_HISTORY = 15` 유지
- [x] 큐 크기 초과 시 오래된 데이터 자동 폐기

### 5. 런타임 조절
- [x] `G` 키: GPU-offload 비율 조절 (GPU 레이어, Temperature, Context 변경)
- [x] `LLM_GPU_LAYERS` 환경변수로 레이어 할당량 동적 제어
- [n/a] GPU 온도/활용률 모니터링 - nvidia-smi 또는 서버 API 필요 (클라이언트에서 자동 throttle 불가)

### 6. 로그 및 디버깅
- [partial] GPU VRAM 사용량 - Llama.cpp 서버가 관리하므로 직접 로깅 불가, 대신 gpu_mode 로그 출력
- [x] 번역 지연시간 (`elapsed` time) 로그 → 이상치 감지
- [x] GPU 미사용 모드 (`LLM_GPU_LAYERS=0`) 시 로그 및 UI 알림

---

## 🔧 권장 설정값

| 파라미터 | 권장값 | 이유 |
|---------|--------|------|
| `temperature` | 0.1 ~ 0.3 | 번역에는 낮은 온도가 안정적 |
| `num_ctx` | 2048 ~ 4096 | 번역 문장 길이에는 충분 |
| `num_gpu` | 0 or 1 | 내장 GPU는 1, 외장 GPU는 0~2 |
| `batch_size` | 1 | 실시간 처리에는 배치 불필요 |
| `translation_semaphore` | 1 | GPU 메모리 보호용 |

---

## 📌 결정 사항 기록 (Context Notes)

### 2026-06-01
- **Llama.cpp Server가 HTTP 기반으로 동작**: GPU 관리 책임이 서버側にあり, 클라이언트에서는 semaphore로 동시 요청 수만 제어하면 됨
- **IGPU (내장 그래픽) 환경 우선**: Intel UHD 같은 내장 GPU는 VRAM이 적으므로 `LLM_GPU_LAYERS=0` (CPU only) 기본값 고려
- **faster-whisper는 CPU 최적화**: `int8` + `CPU` 조합이 전력 효율적으로 가장优り
- **numa/temp 모니터링**: Llama.cpp Server 실행 시 `--numa` 플래그로 설정, GPU 온도는 `nvidia-smi`로 별도 모니터링
- **VRAM 로깅 불가**: 클라이언트(HTTP)에서 Llama.cpp 서버의 VRAM 사용량을 직접 조회할 수 없음. `gpu_mode` 로그로 추적

---

## ✅ 완료 기준

| 기준 | 상태 | 비고 |
|------|------|------|
| 1. GPU 미사용 시 CPU 모드 동작 | ✅ 구현 완료 | `LLM_GPU_LAYERS=0` 설정 시 로그/UI 알림 |
| 2. 동시 번역 2개 이상 시 VRAM 초과 방지 | ✅ 구현 완료 | 세마포어로 동시 1개로 제한 |
| 3. 번역 지연시간 5초 이하 | ⚠️ 런타임 검증 필요 | `elapsed` time 로그로 추적 가능 |
| 4. 로그에 GPU/VRAM 사용량 출력 | ✅ GPU 모드 로깅 | 실제 VRAM은 서버 관리 (직접 로깅 불가) |

---

## 🔑 G键 사용법

```
실행 중:
  G → Llama.cpp 설정 변경 메뉴
       1: GPU 레이어 토글 (0 → 8 → 16 → 0 순환)
       2: Temperature 조정
       3: Context Window 조정
```

## 🔑 환경변수 설정법

```bash
# CPU만 사용 (IGPU 환경)
set LLM_GPU_LAYERS=0

# GPU 사용 (외장 GPU, 레이어 수 지정)
set LLM_GPU_LAYERS=19

# 번역 파라미터 조정
set LLM_TEMPERATURE=0.2
set LLM_NUM_CTX=2048
set LLM_MAX_CONCURRENT=1
set LLAMACPP_HOST=http://localhost:8080
```