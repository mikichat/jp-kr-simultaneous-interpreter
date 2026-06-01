# Goal Prompt: GPU 자원 낭비 방지 및 효율적인 로컬 LLM 번역

## 🎯 목표

로컬 LLM 번역 파이프라인에서 GPU 자원을 낭비하지 않고 효율적으로 동작하도록 최적화한다.

---

## 📋 체크리스트

### 1. Llama.cpp Server 설정 최적화
- [ ] `LLAMACPP_HOST`, `LLAMACPP_MODEL` 환경변수로 분리 (하드코딩 제거)
- [ ] `temperature` 0.3 이하로 고정 (창의성浪费 방지)
- [ ] `num_ctx` (컨텍스트 윈도우)를 필요 최소값으로 설정 (VRAM 절약)
  - 권장: 2048 ~ 4096 (번역 태스크에는 충분)
- [ ] `num_gpu` 설정을 통해 GPU 사용량을 명시적으로 제한
  - 예: `LLAMACPP_NGPU=0` → CPU만 사용 (IGPU 환경에서 필수)
- [ ] `numa` (NUMA亲和성) 활성화로 멀티소켓 메모리 성능 향상

### 2. 번역 워커 병렬 처리 제어
- [ ] `TRANSLATE_WORKERS` 기본값을 `1`로 유지 (LLM은 병렬보다 순차 처리가 안정적)
- [ ] 동시 요청 시 GPU 메모리 초과 방지을 위한 세마포어 구현
  ```python
  max_concurrent_translations = 1  # GPU VRAM 보호
  translation_semaphore = threading.Semaphore(max_concurrent_translations)
  ```
- [ ] `translate_queue.qsize()` 가 임계값 초과 시 새 요청 거부 (메모리 보호)

### 3. STT (Whisper) GPU 사용 안 함
- [ ] `device="cpu"` 고정 (이미 설정됨 - 유지)
- [ ] `compute_type="int8"` 고정 (이미 설정됨 - 유지)
- [ ] 배치 크기 `1`로 고정 (实时 처리에서는 배치 불필요)

### 4. 메모리 관리
- [ ] 오디오 버퍼 크기 명시적 제한 (`MAX_AUDIO_BUFFER_SEC = 30`)
- [ ] 히스토리 최대치 enforcement (`MAX_HISTORY = 15` 이미 설정됨 - 유지)
- [ ] 큐 크기 초과 시 오래된 데이터 자동 폐기 (메모리릭 방지)

### 5. 런타임 조절
- [ ] `M` 키: GPU-offload 비율 조절 (0=CPU only, 1=部分 GPU, 2=Full GPU)
- [ ] 환경변수 `LLM_GPU_LAYERS`로 레이어 할당량 동적 제어
- [ ] GPU 온도/활용률 모니터링 → 임계값 초과 시 자동 throttle

### 6. 로그 및 디버깅
- [ ] GPU VRAM 사용량定期 로그 출력
- [ ] 번역 지연시간 (latency) 로그 → 이상치 감지
- [ ] "GPU 미사용 모드" 전환 시 사용자 알림

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
- **IGPU (내장 그래픽) 환경 우선**: Intel UHD 같은 내장 GPU는 VRAM이 적으므로 `LLAMACPP_NGPU=0` (CPU only) 기본값 고려
- **faster-whisper는 CPU 최적화**: `int8` + `CPU` 조합이 전력 효율적으로 가장优り

---

## ✅ 완료 기준

1. GPU 미사용 시 (IGPU 환경) 오류 없이 CPU 모드로 동작
2. 동시 번역 요청이 2개 이상일 때 GPU VRAM 초과하지 않음
3. 번역 지연시간이 5초 이하 (평균 2~3초 목표)
4. 로그에 GPU/VRAM 사용량 정상적으로 출력