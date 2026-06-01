#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JP/EN→KR 동시통역 - 비동기 파이프라인
- 오디오 캡처: sounddevice
- 음성인식: faster-whisper (로컬, CPU 전용)
- 번역: Llama.cpp Server (GPU 자원 효율化管理)
- UI: rich 터미널

[파이프라인 구조]
  오디오 입력 → [Audio Collector] → stt_queue
                                       ↓
                                  [STT Worker] → translate_queue
                                                      ↓
                                          [Translate Worker × semaphore] → 화면 출력
"""

import threading
import queue
import time
import sys
import signal
import os
from datetime import datetime
from typing import Any
import logging

# ────────── 로깅 설정 ──────────
logging.basicConfig(
    filename='interpreter.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from ollama import Client as OllamaClient
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.prompt import Prompt

# ─────────────────────────────────────────────
# GPU 효율화 설정 (환경변수优先)
# ─────────────────────────────────────────────
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "0"))  # 0=CPU only, 양수=GPU 레이어 수
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 0.1~0.3 권장
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "2048"))  # 번역에는 2048이면 충분
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "1"))  # GPU VRAM 보호용 세마포어
MAX_AUDIO_BUFFER_SEC = 30  # 오디오 버퍼 최대 크기 (메모리 보호)

# ─────────────────────────────────────────────
# Llama.cpp Server 설정 (GPU 효율화)
# ─────────────────────────────────────────────
LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", "http://localhost:8080")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "")  # 서버 시작 시 지정된 모델

SAMPLE_RATE  = 16000          # Whisper 권장 샘플레이트
CHUNK_SEC    = 1              # 오디오 청크 단위 (초)
CHANNELS     = 1              # 모노
WHISPER_MODEL = "small"       # tiny / base / small / medium

MAX_HISTORY   = 15            # 표시할 최대 히스토리 수
STT_WORKERS   = 1             # STT 워커 수 (CPU 기반이라 1개 권장)
TRANSLATE_WORKERS = 1         # 번역 워커 수 (세마포어로 동시 제어)
SOURCE_LANG   = "ja"          # 소스 언어 (ja/en/zh/auto)
AUDIO_GAIN         = 3.0      # 오디오 증폭 배수
SILENCE_MULTIPLIER = 1.05     # 노이즈 플로어 대비 이 배수 이상이면 음성으로 판단
MIN_RMS_THRESHOLD  = 0.0100    # RMS 최소 임계값 (자막음 등 작은 소리 필터링)
VAD_MIN_SILENCE_MS = 500       # VAD 최소 무음 시간 (ms)
VAD_THRESHOLD = 0.5           # VAD 임계값 (0~1)
NOISE_FLOOR_DECAY  = 0.995     # 노이즈 플로어 감소율

# ─────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────
console = Console()

# 파이프라인 큐
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()        # 오디오 원본 청크
stt_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=10)  # STT 대기 오디오
translate_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=20)   # 번역 대기 텍스트

# 스레드 안전 히스토리 잠금
history_lock = threading.Lock()
history: list[dict] = []

current_src = ""   # 현재 인식된 원문 (일본어 또는 영어)
current_kr = ""
is_running = False
is_paused = False
status_msg = "대기 중"
chunk_count = 0
stt_pending = 0       # STT 대기 중인 오디오 수
translate_pending = 0  # 번역 대기 중인 텍스트 수
error_msg = ""
noise_floor = 0.0     # 적응형 노이즈 플로어 (자동 측정)
current_rms = 0.0     # 현재 RMS (UI 표시용)

# 런타임 변경용 전역 변수
current_device_idx: int = 0       # 현재 오디오 장치 인덱스
audio_stream: "sd.InputStream | None" = None  # 현재 오디오 스트림
stream_lock = threading.Lock()  # 스트림 변경용 잠금

# 명령 큐 (런타임 변경 명령)
command_queue: "queue.Queue[str]" = queue.Queue()

# GPU VRAM 보호용 세마포어 (동시 번역 요청 수 제한)
translation_semaphore: "threading.Semaphore" = None  # 런타임에서 초기화

# 오디오 버퍼 플래그 (메모리 보호)
audio_buffer_too_large = False

# ─────────────────────────────────────────────
# 오디오 장치 목록 표시
# ─────────────────────────────────────────────
def list_audio_devices() -> list[dict]:
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            input_devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "hostapi": sd.query_hostapis(d["hostapi"])["name"],
            })
    return input_devices


def test_device_audio(device_idx: int, duration: float = 3.0) -> float:
    """장치에서 3초간 마이크 입력을监听하여 RMS 레벨 측정."""
    try:
        import threading
        level_holder = [0.0]

        def callback(indata, frames, time_info, status):
            if status:
                return
            rms = np.sqrt(np.mean(indata**2))
            if rms > level_holder[0]:
                level_holder[0] = rms

        with sd.InputStream(device=device_idx, channels=CHANNELS,
                           samplerate=SAMPLE_RATE, callback=callback):
            sd.sleep(int(duration * 1000))
        return level_holder[0]
    except Exception:
        return 0.0


def select_device() -> int:
    """사용자가 오디오 장치를 선택합니다."""
    devices = list_audio_devices()

    table = Table(title="🎙️  오디오 입력 장치 목록", box=box.ROUNDED, border_style="cyan")
    table.add_column("번호", style="bold cyan", width=6)
    table.add_column("장치명", style="white")
    table.add_column("Host API", style="dim")
    table.add_column("레벨", style="dim", width=10)

    # WASAPI Loopback 장치 강조 (PC 사운드 캡처용)
    loopback_idx = None
    for d in devices:
        is_loopback = (
            "loopback" in d["name"].lower()
            or "스테레오 믹스" in d["name"]
            or "stereo mix" in d["name"].lower()
            or "what u hear" in d["name"].lower()
        )
        style = "bold green" if is_loopback else ""
        table.add_row(
            str(d["index"]),
            f"{'🔊 ' if is_loopback else ''}{d['name']}",
            d["hostapi"],
            "▓▓▓▓▓▓▓▓▓▓",
            style=style,
        )
        if is_loopback and loopback_idx is None:
            loopback_idx = d["index"]

    console.print(table)
    console.print("[cyan]🔊 각 장치에서 3초간 테스트 사운드 출력 중...[/cyan]")

    # 각 장치 테스트
    device_levels = {}
    for d in devices:
        console.print(f"  [{d['index']}] {d['name'][:40]}... 테스트 중...", end="", style="dim")
        level = test_device_audio(d["index"], duration=3.0)
        device_levels[d["index"]] = level
        bar = "█" * int(level * 30)
        console.print(f" {bar or '·'} ({level:.3f})", style="green" if level > 0.01 else "dim")

    console.print(
        "\n[dim]💡 PC에서 재생되는 소리 캡처: "
        "[bold green]🔊 표시된 Loopback / 스테레오 믹스[/] 장치 선택[/dim]\n"
    )

    default = str(loopback_idx if loopback_idx is not None else devices[0]["index"])
    choice = Prompt.ask(
        f"[cyan]장치 번호를 입력하세요[/]",
        default=default,
    )

    try:
        idx = int(choice)
        # 유효성 검사
        valid_indices = [d["index"] for d in devices]
        if idx not in valid_indices:
            console.print(f"[red]잘못된 장치 번호입니다. 기본값({default}) 사용.[/red]")
            idx = int(default)
        return idx
    except ValueError:
        return int(default)


# ─────────────────────────────────────────────
# 오디오 캡처 콜백
# ─────────────────────────────────────────────
def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    if status:
        pass  # 오버플로 등 무시
    with stream_lock:
        if audio_stream is not None:
            audio_queue.put(indata.copy())


# ─────────────────────────────────────────────
# [파이프라인 1단계] 오디오 수집기
# - audio_queue에서 raw 데이터를 모아 CHUNK_SEC 단위로 stt_queue에 전달
# - 항상 돌면서 오디오를 버퍼링하므로 유실 없음
# - 적응형 노이즈 플로어: 처음 몇 초간 주변 소음을 측정하고
#   그 기준보다 큰 소리만 음성으로 인식
# ─────────────────────────────────────────────
def audio_collector():
    global chunk_count, noise_floor, current_rms

    buffer = []
    samples_per_chunk = SAMPLE_RATE * CHUNK_SEC

    # 노이즈 플로어 측정용
    calibration_rms_list = []
    calibration_done = False
    calibration_chunks = 5  # 처음 5개 청크(약 5초)로 주변 소음 측정 - 더 안정적인 기준선 확보
    calibration_rejected = 0  # 캘리브레이션 중 급격한 변화 건너뜀 횟수

    while is_running:
        try:
            chunk = audio_queue.get(timeout=0.3)

            # 일시 중지 시 오디오 무시
            if is_paused:
                continue
            buffer.append(chunk)

            # 충분한 오디오가 쌓이면 처리
            total_samples = sum(c.shape[0] for c in buffer)
            if total_samples >= samples_per_chunk:
                # 버퍼 합치기
                audio_data = np.concatenate(buffer, axis=0).flatten().astype(np.float32)
                buffer = []  # 버퍼 비우기 (단어마다 바로 처리)

                # 오디오 버퍼 크기 명시적 제한 (메모리 보호)
                buffer_sec = len(audio_data) / SAMPLE_RATE
                if buffer_sec > MAX_AUDIO_BUFFER_SEC:
                    audio_buffer_too_large = True
                    logging.warning(f"[Audio] 버퍼 크기 초과 ({buffer_sec:.1f}초 > {MAX_AUDIO_BUFFER_SEC}초) →_truncate")
                    audio_data = audio_data[int(-MAX_AUDIO_BUFFER_SEC * SAMPLE_RATE):]  # 최근 것만 유지
                    audio_buffer_too_large = False

                # 오디오 증폭 (AUDIO_GAIN 적용)
                if AUDIO_GAIN != 1.0:
                    audio_data = audio_data * AUDIO_GAIN
                    # 클리핑 방지 (-1.0 ~ 1.0 범위로 제한)
                    audio_data = np.clip(audio_data, -1.0, 1.0)

                rms = float(np.sqrt(np.mean(audio_data ** 2)))
                current_rms = rms

                # 캘리브레이션: 처음 몇 청크로 노이즈 플로어 측정 (안정적인 기준선 확보)
                if not calibration_done:
                    # 급격한 RMS 변화 감지 (자막음처럼 평소와 다른 소리는 건너뜀)
                    if len(calibration_rms_list) > 0:
                        prev_rms = calibration_rms_list[-1]
                        # 이전 값과 2배 이상 차이나면 자막음 등으로 판단하여 무시
                        if rms > prev_rms * 2.0 or rms < prev_rms * 0.5:
                            calibration_rejected += 1
                            logging.info(f"[캘리브레이션] 이상치 건너뜀 ({calibration_rejected}/5): RMS={rms:.6f}")
                            if calibration_rejected >= 5:
                                # 너무 많이 건너뛰면 현재 값을 기준선으로 설정
                                noise_floor = rms
                                calibration_done = True
                                logging.info(f"[캘리브레이션 완료] 이상치 과다 → 현재 RMS: {noise_floor:.6f}")
                            continue

                    calibration_rms_list.append(rms)
                    logging.info(f"[캘리브레이션] RMS 측정 중... ({len(calibration_rms_list)}/{calibration_chunks}) RMS={rms:.6f}")
                    if len(calibration_rms_list) >= calibration_chunks:
                        noise_floor = max(np.mean(calibration_rms_list), MIN_RMS_THRESHOLD)
                        calibration_done = True
                        logging.info(f"[캘리브레이션 완료] 노이즈 플로어: {noise_floor:.6f}, 음성 임계값: {noise_floor * SILENCE_MULTIPLIER:.6f}")
                    continue

                # 적응형 노이즈 플로어 업데이트 (환경 변화 대응)
                if rms < noise_floor * SILENCE_MULTIPLIER * 0.8:
                    noise_floor = noise_floor * NOISE_FLOOR_DECAY + rms * (1 - NOISE_FLOOR_DECAY)
                    noise_floor = max(noise_floor, MIN_RMS_THRESHOLD)
                elif rms > noise_floor * 10:
                    pass  # 순간 소음 무시

                # 적응형 무음 감지
                threshold = max(noise_floor * SILENCE_MULTIPLIER, MIN_RMS_THRESHOLD)
                if rms < threshold:
                    logging.debug(f"[소리입력] 무음 (RMS: {rms:.6f} < 임계값: {threshold:.6f})")
                    continue

                chunk_count += 1

                # stt_queue에 넣기 (가득 차면 가장 오래된 것 버림)
                try:
                    stt_queue.put_nowait(audio_data)
                    logging.info(f"[Audio] 청크 #{chunk_count} → STT 큐 전달 (대기: {stt_queue.qsize()})")
                except queue.Full:
                    try:
                        stt_queue.get_nowait()
                    except queue.Empty:
                        pass
                    stt_queue.put_nowait(audio_data)
                    logging.warning(f"[Audio] STT 큐 가득참 → 오래된 청크 제거 후 #{chunk_count} 추가")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[Audio Collector 오류] {e}")


# ─────────────────────────────────────────────
# [런타임 변경] 오디오 장치 변경
# ─────────────────────────────────────────────
def change_audio_device(new_device_idx: int) -> bool:
    """오디오 장치를 변경하고 스트림을 다시 시작합니다. 번역 중에도 호출 가능."""
    global current_device_idx, audio_stream

    try:
        dev_info = sd.query_devices(new_device_idx)
        device_name = dev_info['name']
        logging.info(f"[장치 변경] {current_device_idx} → {new_device_idx} ({device_name})")

        # 스트림 잠금으로 콜백 보호しながら 장치 교체
        with stream_lock:
            # 기존 스트림 닫기 (콜백 실행 중이면 완료까지 대기)
            if audio_stream is not None:
                old_stream = audio_stream
                audio_stream = None  # 먼저 None 설정하여 콜백 무효화
                old_stream.close()
                logging.info("[장치 변경] 기존 스트림 닫기 완료")

            # 새 스트림 생성
            audio_stream = sd.InputStream(
                device=new_device_idx,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1),
            )
            audio_stream.start()
            current_device_idx = new_device_idx
            logging.info(f"[장치 변경] 새 스트림 시작: {device_name}")

        console.print(f"[green]✓ 오디오 장치 변경 완료: {device_name}[/green]")
        logging.info(f"[장치 변경 완료] {device_name}")
        return True
    except Exception as e:
        logging.error(f"[장치 변경 실패] {e}")
        console.print(f"[red]✗ 장치 변경 실패: {e}[/red]")
        # 실패 시에도 스트림 복원 시도
        try:
            with stream_lock:
                if audio_stream is None:
                    audio_stream = sd.InputStream(
                        device=current_device_idx,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        callback=audio_callback,
                        blocksize=int(SAMPLE_RATE * 0.1),
                    )
                    audio_stream.start()
        except:
            pass
        return False


# ─────────────────────────────────────────────
# [런타임 변경] GPU 모드 확인/조절
# ─────────────────────────────────────────────
def get_gpu_mode_display() -> str:
    """현재 GPU 모드를 문자열로 반환"""
    if LLM_GPU_LAYERS == 0:
        return "CPU only"
    else:
        return f"GPU ({LLM_GPU_LAYERS} layers)"


def change_llamacpp_settings():
    """Llama.cpp 설정 변경 (G键)"""
    global LLM_GPU_LAYERS, LLM_TEMPERATURE, LLM_NUM_CTX

    console.print("\n[cyan]═══ Llama.cpp 설정 변경 ═══[/cyan]")
    gpu_mode = get_gpu_mode_display()

    table = Table(title="현재 설정", box=box.ROUNDED, border_style="cyan")
    table.add_column("항목", style="bold cyan", width=20)
    table.add_column("값", style="white")
    table.add_row("GPU 모드", f"{gpu_mode} (G键으로 토글)")
    table.add_row("Temperature", str(LLM_TEMPERATURE))
    table.add_row("Context Window", str(LLM_NUM_CTX))
    table.add_row("동시 번역 제한", str(LLM_MAX_CONCURRENT))
    console.print(table)

    console.print("\n[dim]G键: GPU 레이어 토글 (0 → 8 → 16 → 0 순환)[/dim]")
    choice = Prompt.ask(
        "[cyan]변경할 설정 번호를 입력하세요 (취소: Enter)[/cyan]",
        default="",
        choices=["1", "2", "3"],
    )

    if choice == "1":
        # GPU 레이어 조절 (0 → 8 → 16 → 0 순환)
        new_layers = (LLM_GPU_LAYERS + 8) % 24
        console.print(f"[yellow]GPU 레이어 변경: {LLM_GPU_LAYERS} → {new_layers}[/yellow]")
        console.print(f"[dim]※ Llama.cpp Server 재시작 필요 (Ctrl+C 후 server 재실행)[/dim]")
        logging.info(f"[설정 변경] GPU layers: {LLM_GPU_LAYERS} → {new_layers} (재시작 필요)")
    elif choice == "2":
        new_temp = Prompt.ask("[cyan]Temperature 값 (0.1~0.3)[/cyan]", default=str(LLM_TEMPERATURE))
        try:
            new_temp_f = float(new_temp)
            if 0.0 < new_temp_f <= 1.0:
                LLM_TEMPERATURE = new_temp_f
                console.print(f"[green]✓ Temperature: {LLM_TEMPERATURE}[/green]")
                logging.info(f"[설정 변경] Temperature: {new_temp_f}")
        except ValueError:
            console.print("[red]✗ 잘못된 값입니다.[/red]")
    elif choice == "3":
        new_ctx = Prompt.ask("[cyan]Context Window 크기 (512~8192)[/cyan]", default=str(LLM_NUM_CTX))
        try:
            new_ctx_i = int(new_ctx)
            if 512 <= new_ctx_i <= 8192:
                LLM_NUM_CTX = new_ctx_i
                console.print(f"[green]✓ Context: {LLM_NUM_CTX}[/green]")
                logging.info(f"[설정 변경] num_ctx: {new_ctx_i}")
        except ValueError:
            console.print("[red]✗ 잘못된 값입니다.[/red]")

    console.print("[cyan]═════════════════════════[/cyan]\n")


# ─────────────────────────────────────────────
# [런타임 변경] 명령 핸들러 스레드
# ─────────────────────────────────────────────
def command_handler():
    """메인 루프와 별개로 사용자 명령을 처리합니다. D=장치변경, G=GPU설정"""
    global is_running

    import readchar

    while is_running:
        try:
            # readchar은 screen mode에서도 키 입력을 감지합니다
            key = readchar.readchar()
            if key in ('q', 'Q'):
                is_running = False
                break
            elif key == readchar.key.ESC:
                # ESC 키: 일시 중지/재개 토글
                global is_paused
                is_paused = not is_paused
                if is_paused:
                    console.print("\n[yellow]⏸  일시 중지됨 (ESC 다시 누르면 재개)[/yellow]")
                    logging.info("[사용자] 일시 중지")
                else:
                    console.print("\n[green]▶  재개[/green]")
                    logging.info("[사용자] 재개")
            elif key in ('d', 'D'):
                # 장치 변경 요청
                command_queue.put("change_device")
            elif key in ('g', 'G'):
                # GPU 설정 변경 요청
                command_queue.put("change_gpu")
        except Exception:
            pass

        # command_queue에서 명령 처리
        try:
            cmd = command_queue.get(timeout=0.1)
            if cmd == "change_device":
                handle_device_change()
            elif cmd == "change_gpu":
                change_llamacpp_settings()
        except queue.Empty:
            pass


def handle_device_change():
    """오디오 장치 변경 처리 (Live UI 일시 중단 후 실행)"""
    global is_running

    # Live UI가 처리 중인지 확인하고 잠시 대기
    console.print("\n[cyan]═══ 오디오 장치 변경 ═══[/cyan]")
    devices = list_audio_devices()

    table = Table(title="🎙️  오디오 입력 장치 목록", box=box.ROUNDED, border_style="cyan")
    table.add_column("번호", style="bold cyan", width=6)
    table.add_column("장치명", style="white")
    table.add_column("Host API", style="dim")

    for d in devices:
        is_current = d["index"] == current_device_idx
        style = "bold green" if is_current else ""
        marker = " ◀현재" if is_current else ""
        table.add_row(
            str(d["index"]),
            f"{d['name']}{marker}",
            d["hostapi"],
            style=style,
        )

    console.print(table)
    choice = Prompt.ask(
        "[cyan]새 장치 번호를 입력하세요 (취소: Enter)[/cyan]",
        default="",
    )

    if choice.strip():
        try:
            new_idx = int(choice.strip())
            change_audio_device(new_idx)
        except ValueError:
            console.print("[red]✗ 잘못된 번호입니다.[/red]")
    console.print("[cyan]═════════════════════════[/cyan]\n")


# ─────────────────────────────────────────────
# [파이프라인 2단계] STT 워커
# - stt_queue에서 오디오를 꺼내 Whisper로 텍스트 변환
# - 결과를 translate_queue에 전달
# ─────────────────────────────────────────────
def stt_worker(whisper: WhisperModel, worker_id: int):
    global current_jp, status_msg, stt_pending, error_msg

    while is_running:
        try:
            audio_data = stt_queue.get(timeout=0.5)
            stt_pending = stt_queue.qsize()
            status_msg = "🎙️  음성인식 중..."

            try:
                lang_param = SOURCE_LANG if SOURCE_LANG != "auto" else None
                lang_label = {"ja": "일본어", "en": "영어", "zh": "중국어"}.get(SOURCE_LANG, "자동감지")
                logging.info(f"[STT-{worker_id}] Whisper 음성인식 시작... (언어: {lang_label})")
                start_time = time.time()
                transcribe_kwargs = {
                    "beam_size": 3,
                    "vad_filter": True,
                    "vad_parameters": {
                        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                        "threshold": VAD_THRESHOLD,  # VAD 임계값 (0~1, 낮을수록 민감)
                    },
                }
                if lang_param:
                    transcribe_kwargs["language"] = lang_param
                segments, info = whisper.transcribe(audio_data, **transcribe_kwargs)
                src_text = " ".join(s.text.strip() for s in segments).strip()
                detected_lang = getattr(info, 'language', SOURCE_LANG)
                elapsed = time.time() - start_time
                logging.info(f"[STT-{worker_id}] 인식결과 ({elapsed:.1f}초) [감지언어: {detected_lang}]: {src_text}")
            except Exception as e:
                error_msg = f"STT 오류: {e}"
                logging.error(f"[STT-{worker_id} 오류] {e}", exc_info=True)
                status_msg = "대기 중"
                continue

            if not src_text:
                logging.info(f"[STT-{worker_id}] 텍스트가 비어있음 (스킵)")
                status_msg = "대기 중"
                continue

            current_src = src_text

            # translate_queue에 넣기 (감지된 언어 정보도 함께)
            item = {"text": src_text, "lang": detected_lang}
            try:
                translate_queue.put_nowait(item)
                logging.info(f"[STT-{worker_id}] → 번역 큐 전달 (대기: {translate_queue.qsize()})")
            except queue.Full:
                logging.warning(f"[STT-{worker_id}] 번역 큐 가득참, 최신 텍스트로 교체")
                try:
                    translate_queue.get_nowait()
                except queue.Empty:
                    pass
                translate_queue.put_nowait(item)

            status_msg = "대기 중"

        except queue.Empty:
            stt_pending = 0
            continue
        except Exception as e:
            error_msg = f"STT 오류: {e}"
            logging.error(f"[STT Worker 오류] {e}")
            status_msg = "대기 중"


# ─────────────────────────────────────────────
# [파이프라인 3단계] 번역 워커
# - translate_queue에서 소스 텍스트를 꺼내 Ollama로 번역
# - 여러 워커가 병렬로 번역 처리 가능
# ─────────────────────────────────────────────
def _build_prompt(src_text: str, src_lang: str) -> str:
    """간단한 실시간 번역 프롬프트를 생성합니다."""
    lang_name = {"ja": "일본어", "en": "영어", "zh": "중국어"}.get(src_lang, "외국어")

    prompt = (
        f"당신은 실시간 동시통역사입니다. {lang_name}를 자연스러운 한국어로 번역해주세요.\n"
        f"존댓말(~합니다, ~니다)을 사용하고, 결과만 출력하세요:\n\n{src_text}"
    )

    return prompt




def translate_with_llamacpp(prompt: str) -> str:
    """Llama.cpp Server를 사용하여 번역 수행 (GPU 효율화: 스트리밍, OpenAI 호환)"""
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLAMACPP_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": LLM_TEMPERATURE,  # 환경변수에서 설정 (0.1~0.3)
        "num_ctx": LLM_NUM_CTX,  # VRAM 절약용 컨텍스트 윈도우 크기
    }

    kr_text = ""
    current_kr = ""
    gpu_mode = "CPU" if LLM_GPU_LAYERS == 0 else f"GPU({LLM_GPU_LAYERS} layers)"
    logging.info(f"[Llama.cpp] 요청 중... (mode: {gpu_mode}, temp: {LLM_TEMPERATURE}, ctx: {LLM_NUM_CTX})")
    with httpx.stream("POST", f"{LLAMACPP_HOST}/v1/chat/completions", headers=headers, json=payload, timeout=120.0) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.strip()
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                line = line[6:].strip()
            if not line or line == "[DONE]":
                break
            import json
            try:
                data = json.loads(line)
                if data.get("choices"):
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        kr_text += delta
                        current_kr = kr_text
            except json.JSONDecodeError:
                continue
    return kr_text


def translate_worker(_worker_id: int):
    global current_kr, status_msg, translate_pending, error_msg

    while is_running:
        try:
            # 세마포어로 GPU VRAM 보호 (동시 번역 수 제한)
            with translation_semaphore:
                item = translate_queue.get(timeout=0.5)
                src_text = item["text"]
                src_lang = item.get("lang", SOURCE_LANG)
                translate_pending = translate_queue.qsize()
                status_msg = "🔄  번역 중..."

                # 큐 임계값 초과 시 새 요청 거부 (메모리 보호)
                if translate_pending > 15:
                    logging.warning(f"[번역-{_worker_id}] 큐 임계값 초과 ({translate_pending}/20) → 스킵")
                    continue

                try:
                    prompt = _build_prompt(src_text, src_lang)
                    logging.info(f"[번역-{_worker_id}] 요청 중... (백엔드: Llama.cpp, 언어: {src_lang})")
                    logging.debug(f"[번역-{_worker_id}] 프롬프트:\n{prompt}")
                    start_time = time.time()

                    # 스트리밍 응답으로 실시간 번역 결과 표시
                    kr_text = ""
                    current_kr = ""  # 초기화

                    for chunk in translate_with_llamacpp(prompt):
                        kr_text += chunk
                        current_kr = kr_text

                    elapsed = time.time() - start_time
                    logging.info(f"[번역-{_worker_id}] 완료 ({elapsed:.1f}초): {kr_text}")
                    current_kr = kr_text
                    error_msg = ""

                    # 히스토리 추가 (스레드 안전)
                    lang_flag = {"ja": "🇯🇵", "en": "🇺🇸", "zh": "🇨🇳"}.get(src_lang, "🌐")
                    with history_lock:
                        history.insert(0, {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "src": src_text,
                            "kr": kr_text,
                            "lang": src_lang,
                            "flag": lang_flag,
                        })
                        if len(history) > MAX_HISTORY:
                            history.pop()

                    status_msg = "✅  번역 완료"
                    time.sleep(0.3)
                    status_msg = "대기 중"
                except Exception as e:
                    error_msg = f"번역 오류: {e}"
                    logging.error(f"[번역-{_worker_id} 오류] {e}")
                    status_msg = "대기 중"
                    continue

        except queue.Empty:
            translate_pending = 0
            continue
        except Exception as e:
            error_msg = f"오류: {e}"
            logging.error(f"[Translate Worker 오류] {e}")
            status_msg = "대기 중"


# ─────────────────────────────────────────────
# 터미널 UI 렌더링
# ─────────────────────────────────────────────
def build_ui() -> Layout:
    global status_msg
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="current", size=8),
        Layout(name="history"),
        Layout(name="footer", size=3),
    )

    # 헤더 - 소스 언어 표시
    lang_display = {"ja": "JP", "en": "EN", "zh": "CN", "auto": "AUTO"}.get(SOURCE_LANG, "?")
    lang_flag = {"ja": "🎌", "en": "🇺🇸", "zh": "🇨🇳", "auto": "🌐"}.get(SOURCE_LANG, "🌐")
    backend_name = "Llama.cpp"
    header_text = Text()
    header_text.append(f"{lang_flag} ", style="bold red")
    header_text.append(f"{lang_display} → KR ", style="bold white")
    header_text.append("동시통역 ", style="bold yellow")
    header_text.append(f"| {backend_name}", style="dim")
    header_text.append(f" | 실시간 번역 (STT×{STT_WORKERS} / 번역×{TRANSLATE_WORKERS})", style="dim")
    layout["header"].update(
        Panel(Align.center(header_text), border_style="bright_blue", padding=(0, 2))
    )

    # 현재 번역 결과
    src_title_map = {"ja": "[bold red]🇯🇵 日本語[/bold red]", "en": "[bold green]🇺🇸 English[/bold green]", "zh": "[bold yellow]🇨🇳 中文[/bold yellow]", "auto": "[bold yellow]🌐 원문 (자동감지)[/bold yellow]"}
    src_style_map = {"ja": "red", "en": "green", "zh": "yellow", "auto": "yellow"}
    src_panel = Panel(
        Text(current_src or "[dim]음성 대기 중...[/dim]", no_wrap=False),
        title=src_title_map.get(SOURCE_LANG, "[bold]원문[/bold]"),
        border_style=src_style_map.get(SOURCE_LANG, "white"),
        padding=(0, 1),
    )
    kr_panel = Panel(
        Text(current_kr or "[dim]번역 결과 대기 중...[/dim]", no_wrap=False),
        title="[bold blue]🇰🇷 한국어[/bold blue]",
        border_style="blue",
        padding=(0, 1),
    )
    layout["current"].update(
        Layout(Columns([src_panel, kr_panel], equal=True, expand=True))
    )

    # 히스토리
    hist_table = Table(
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        expand=True,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    hist_table.add_column("시각", style="dim", width=10, no_wrap=True)
    hist_table.add_column("언어", style="dim", width=4, no_wrap=True)
    hist_table.add_column("원문", ratio=2)
    hist_table.add_column("한국어", style="blue", ratio=3)

    with history_lock:
        for item in history:
            flag = item.get("flag", "🌐")
            lang = item.get("lang", "?")
            src_style = "red" if lang == "ja" else ("green" if lang == "en" else ("yellow" if lang == "zh" else "white"))
            hist_table.add_row(
                item["time"],
                flag,
                Text(item["src"], style=src_style),
                item["kr"],
            )

    layout["history"].update(
        Panel(
            hist_table if history else Align.center(Text("[dim]번역 히스토리가 없습니다[/dim]")),
            title="[dim]📜 번역 히스토리[/dim]",
            border_style="dim",
        )
    )

    # 푸터 / 상태바
    status_style = "green" if "완료" in status_msg else ("yellow" if "중" in status_msg else "dim")
    if is_paused:
        status_style = "bold yellow"
        status_msg = "⏸  일시 중지됨 (ESC: 재개)"
    threshold = max(noise_floor * SILENCE_MULTIPLIER, MIN_RMS_THRESHOLD)
    rms_style = "green" if current_rms > threshold else "dim"
    status_line = Text()
    status_line.append(f" {status_msg}", style=status_style)
    status_line.append(f"  |  🔊 RMS: {current_rms:.4f}", style=rms_style)
    status_line.append(f" (임계: {threshold:.4f})", style="dim")
    status_line.append(f"  |  청크: {chunk_count}", style="dim")
    status_line.append(f"  |  STT: {stt_pending}", style="cyan")
    status_line.append(f"  |  번역: {translate_pending}", style="magenta")
    status_line.append(f"  |  장치: [yellow]{current_device_idx}[/yellow]", style="dim")
    if error_msg:
        status_line.append(f"  ⚠️  {error_msg}", style="bold red")
    status_line.append("  |  [ESC]일시중지  [D]장치  [M]모델  |  종료: [bold]Ctrl+C[/bold]", style="dim")

    layout["footer"].update(
        Panel(status_line, border_style="dim", padding=(0, 1))
    )

    return layout


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def select_language() -> str:
    """소스 언어를 선택합니다."""
    console.print("\n[cyan]소스 언어를 선택하세요:[/cyan]")
    console.print("  [bold]1[/bold]  🎌  일본어 (JP → KR)")
    console.print("  [bold]2[/bold]  🇺🇸  영어   (EN → KR)")
    console.print("  [bold]3[/bold]  🇨🇳  중국어 (CN → KR)")
    console.print("  [bold]4[/bold]  🌐  자동감지 (AUTO → KR)")
    console.print()
    choice = Prompt.ask(
        "[cyan]번호를 입력하세요[/cyan]",
        default="1",
        choices=["1", "2", "3", "4"],
    )
    lang_map = {"1": "ja", "2": "en", "3": "zh", "4": "auto"}
    return lang_map[choice]




def main():
    global is_running, SOURCE_LANG

    console.print(
        Panel(
            "[bold cyan]JP / EN / CN → KR 동시통역기[/bold cyan]\n"
            "[dim]로컬 Whisper + Llama.cpp Server 기반 실시간 번역[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # Llama.cpp Server 연결 확인
    console.print("\n[cyan]1/4  Llama.cpp Server 연결 확인 중...[/cyan]")
    try:
        with httpx.Client() as client:
            resp = client.get(f"{LLAMACPP_HOST}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                models_data = resp.json()
                model_names = [m.get("id", "") for m in models_data.get("data", []) if m.get("id")]
                console.print(f"[green]✓ 사용 가능한 모델: {', '.join(model_names) or '알 수 없음'}[/green]")
                if model_names:
                    LLAMACPP_MODEL = model_names[0]
                    console.print(f"[green]✓ 첫 번째 모델 사용: {LLAMACPP_MODEL}[/green]")
            else:
                console.print(f"[yellow]⚠️  모델 목록 조회 실패 (상태코드: {resp.status_code})[/yellow]")
                console.print("[dim]  GGUF 모델이 서버에 로드되어 있는지 확인하세요.[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Llama.cpp Server 연결 실패: {e}[/red]")
        console.print(f"[dim]  서버가 {LLAMACPP_HOST} 에서 실행 중인지 확인하세요.[/dim]")
        sys.exit(1)

    # 2. Whisper 모델 로드
    console.print("\n[cyan]2/4  Whisper 모델 로드 중 ({WHISPER_MODEL})...[/cyan]")
    try:
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        console.print("[green]✓ Whisper 모델 로드 완료[/green]")
    except Exception as e:
        console.print(f"[red]✗ Whisper 로드 실패: {e}[/red]")
        sys.exit(1)

    # 3. 소스 언어 선택
    console.print("\n[cyan]3/4  소스 언어 선택[/cyan]")
    SOURCE_LANG = select_language()
    lang_display = {"ja": "일본어 (JP)", "en": "영어 (EN)", "zh": "중국어 (CN)", "auto": "자동감지 (AUTO)"}.get(SOURCE_LANG, "?")
    console.print(f"[green]✓ 소스 언어: {lang_display}[/green]")

    # 4. 오디오 장치 선택
    console.print("\n[cyan]4/4  오디오 장치 선택[/cyan]")
    device_idx = select_device()
    dev_info = sd.query_devices(device_idx)
    device_name = dev_info['name']
    console.print(f"[green]✓ 선택된 장치: {device_name}[/green]\n")

    # 전역 변수 설정 (런타임 변경용)
    global current_device_idx
    current_device_idx = device_idx

    # GPU 효율화: 세마포어 초기화 (동시 번역 수 제한)
    global translation_semaphore
    translation_semaphore = threading.Semaphore(LLM_MAX_CONCURRENT)

    gpu_mode = "CPU only" if LLM_GPU_LAYERS == 0 else f"GPU ({LLM_GPU_LAYERS} layers)"
    logging.info(f"--- 프로그램 시작 (GPU 효율화) ---")
    logging.info(f"번역 백엔드: Llama.cpp Server")
    logging.info(f"GPU 모드: {gpu_mode}, Temperature: {LLM_TEMPERATURE}, Context: {LLM_NUM_CTX}")
    logging.info(f"동시 번역 제한: {LLM_MAX_CONCURRENT} (VRAM 보호)")
    logging.info(f"소스 언어: {lang_display}")
    logging.info(f"선택된 오디오 장치: [{device_idx}] {device_name}")
    logging.info(f"STT 워커: {STT_WORKERS}개, 번역 워커: {TRANSLATE_WORKERS}개")

    # GPU 미사용 모드 알림
    if LLM_GPU_LAYERS == 0:
        console.print("[dim]ℹ️  GPU 미사용 모드 (LLM_GPU_LAYERS=0, CPU only)[/dim]")

    # 시작
    is_running = True

    # ── 파이프라인 워커 스레드 시작 ──

    # 1단계: 오디오 수집기
    collector = threading.Thread(
        target=audio_collector,
        daemon=True,
        name="AudioCollector",
    )
    collector.start()

    # 2단계: STT 워커
    stt_threads = []
    for i in range(STT_WORKERS):
        t = threading.Thread(
            target=stt_worker,
            args=(whisper, i),
            daemon=True,
            name=f"STT-{i}",
        )
        t.start()
        stt_threads.append(t)

    # 3단계: 번역 워커 (병렬) - 전역 ollama_client 사용
    translate_threads = []
    for i in range(TRANSLATE_WORKERS):
        t = threading.Thread(
            target=translate_worker,
            args=(i,),
            daemon=True,
            name=f"Translate-{i}",
        )
        t.start()
        translate_threads.append(t)

    console.print(
        f"[green]✓ 파이프라인 시작: "
        f"오디오 수집 ×1 → STT ×{STT_WORKERS} → 번역 ×{TRANSLATE_WORKERS}[/green]\n"
    )

    # Ctrl+C 핸들러
    def stop(sig, frame):
        global is_running
        is_running = False
        console.print("\n[yellow]⏹  종료 중...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)

    # 4단계: 명령 핸들러 스레드 (D=장치변경, M=모델변경)
    cmd_thread = threading.Thread(
        target=command_handler,
        daemon=True,
        name="CommandHandler",
    )
    cmd_thread.start()

    # 오디오 스트림 + Live UI
    try:
        global audio_stream
        audio_stream = sd.InputStream(
            device=device_idx,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
        )
        audio_stream.start()

        with Live(build_ui(), refresh_per_second=4, screen=True, console=console) as live:
            while is_running:
                # command_queue에 장치/모델 변경 요청이 있는지 확인
                try:
                    cmd = command_queue.get_nowait()
                    # Live 일시 중단 후 처리
                    live.stop()
                    if cmd == "change_device":
                        handle_device_change()
                    elif cmd == "change_model":
                        handle_model_change()
                    # Live 재개
                    live.start()
                except queue.Empty:
                    pass

                live.update(build_ui())
                time.sleep(0.25)

        audio_stream.close()
        audio_stream = None
    except sd.PortAudioError as e:
        console.print(f"[red]오디오 스트림 오류: {e}[/red]")
        if "Invalid sample rate" in str(e):
            console.print(
                "\n[yellow]⚠️  샘플레이트 에러가 발생했습니다.[/yellow]\n"
                "[white]현재 선택하신 장치(예: WASAPI)가 16000Hz 주파수 변환을 지원하지 않아 발생한 문제입니다.\n"
                "다시 실행하셔서 [bold green]1번 (MME)[/bold green] 이나 [bold green]9번 (DirectSound)[/bold green] 등 "
                "다른 '스테레오 믹스' 장치를 선택해 보세요![/white]\n"
            )
        else:
            console.print(
                "[dim]WASAPI Loopback 장치가 없다면 "
                "Windows 사운드 설정 → 녹음 탭 → 스테레오 믹스 활성화[/dim]"
            )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
