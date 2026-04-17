#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JP/EN→KR 동시통역 - 비동기 파이프라인
- 오디오 캡처: sounddevice
- 음성인식: faster-whisper (로컬)
- 번역: Ollama 로컬 LLM 또는 Minimax API
- UI: rich 터미널

[파이프라인 구조]
  오디오 입력 → [Audio Collector] → stt_queue
                                       ↓
                                  [STT Worker] → translate_queue
                                                      ↓
                                                [Translate Worker] → 화면 출력
"""

import threading
import queue
import time
import sys
import signal
from datetime import datetime
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
# 설정
# ─────────────────────────────────────────────
SAMPLE_RATE  = 16000          # Whisper 권장 샘플레이트
CHUNK_SEC    = 3              # 오디오 청크 단위 (초) - 긴 문장 처리를 위해 2→3
CHANNELS     = 1              # 모노
WHISPER_MODEL = "small"       # tiny / base / small / medium
OLLAMA_MODEL  = "aya-expanse:8b"  # aya-expanse:8b / gemma4:e4b
OLLAMA_HOST   = "http://localhost:11434"

# Minimax API 설정
USE_MINIMAX = False           # True로 설정하면 Minimax API 사용
MINIMAX_API_KEY = ""          # Minimax API 키 (환경변수 MINIMAX_API_KEY也可)
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "Minimax-Text-01"  # Minimax 모델명

MAX_HISTORY   = 15            # 표시할 최대 히스토리 수
CONTEXT_HISTORY = 2           # 번역 시 문맥으로 참조할 최근 히스토리 수
STT_WORKERS   = 2             # STT 워커 수 (CPU 기반이라 1개 권장)
TRANSLATE_WORKERS = 2         # 번역 워커 수 (Ollama/Minimax 응답 대기 동안 병렬 처리)
SOURCE_LANG   = "ja"          # 소스 언어 (ja/en/auto)
AUDIO_GAIN         = 4.0      # 오디오 증폭 배수 (1.0=원본, 2.0=2배 증폭, 3.0=3배, 4.0=4배)
SILENCE_MULTIPLIER = 1.5      # 노이즈 플로어 대비 이 배수 이상이면 음성으로 판단 (2.0→1.8: 더 민감)
MIN_RMS_THRESHOLD  = 0.0003   # RMS 최소 임계값 (이 아래는 무조건 무음) - 0.0005→0.0003으로 하향
VAD_MIN_SILENCE_MS = 500       # VAD 최소 무음 시간 (ms) - 200→500로 증가하여 짧은 음성 보호
NOISE_FLOOR_DECAY  = 0.995     # 노이즈 플로어 감소율 (환경 변화 적응용, 0~1)

# ─────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────
console = Console()

# 파이프라인 큐
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()        # 오디오 원본 청크
stt_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=10)  # STT 대기 오디오
translate_queue: "queue.Queue[str]" = queue.Queue(maxsize=20)   # 번역 대기 텍스트

# 스레드 안전 히스토리 잠금
history_lock = threading.Lock()
history: list[dict] = []

current_src = ""   # 현재 인식된 원문 (일본어 또는 영어)
current_kr = ""
is_running = False
status_msg = "대기 중"
chunk_count = 0
stt_pending = 0       # STT 대기 중인 오디오 수
translate_pending = 0  # 번역 대기 중인 텍스트 수
error_msg = ""
noise_floor = 0.0     # 적응형 노이즈 플로어 (자동 측정)
current_rms = 0.0     # 현재 RMS (UI 표시용)

# ─────────────────────────────────────────────
# [문맥 관리] 대화 맥락 추적기
# - 이전 대화에서 언급된 핵심 용어/이름/주제를 기억
# - 시제, 상황, 화자 관계를 추적하여 자연스러운 번역 지원
# ─────────────────────────────────────────────
class ConversationContext:
    """대화 맥락을 추적하여 문장 간 연관성을 고려한 번역 지원"""

    def __init__(self, max_terms: int = 20):
        self.max_terms = max_terms
        self.terms: dict[str, str] = {}  # 용어: 원어 → 번역어 (고정)
        self.topic: str = ""             # 현재 대화 주제
        self.last_speaker: str = ""       # 마지막 발언자 (省略 analyser)
        self.topic_keywords: list[str] = []  # 주제 관련 키워드
        self.sentence_count: int = 0      # 총 대화 수 (시점 판단용)

    def add_translation(self, src_text: str, kr_text: str, src_lang: str):
        """새 번역 결과를 바탕으로 맥락 업데이트"""
        self.sentence_count += 1

        # 핵심 고유 명사 추출 (단어 길이 2~6, 숫자/기호 없음)
        import re
        words = re.findall(r'[가-힣]{2,6}|[A-Z][a-z]{1,20}|[A-Z]{2,}', src_text)
        for word in words:
            if word not in self.terms:
                # 해당 단어의 한국어 번역 찾기
                kr_words = re.findall(r'[가-힣]{2,6}', kr_text)
                for kr_word in kr_words:
                    if len(kr_word) == len(word) or len(kr_word) >= 2:
                        self.terms[word] = kr_word
                        break
            if len(self.terms) > self.max_terms:
                # 가장 오래된 용어 제거
                self.terms.pop(next(iter(self.terms)))

        # 주제 키워드 업데이트 (명사 위주)
        nouns = re.findall(r'[가-힣]+/[nrv]', '') or re.findall(r'[가-힣]{2,6}', src_text)[:3]
        if nouns:
            self.topic_keywords = nouns[:5]

    def build_context_summary(self) -> str:
        """현재 대화 맥락 요약 생성 (프롬프트용)"""
        parts = []

        if self.terms:
            term_list = ", ".join(f"{k}→{v}" for k, v in list(self.terms.items())[-10:])
            parts.append(f"[핵심 용어] {term_list}")

        if self.topic_keywords:
            parts.append(f"[주제 키워드] {', '.join(self.topic_keywords)}")

        if self.sentence_count > 0:
            parts.append(f"[대화 진행] {self.sentence_count}번째 발화")

        return "\n".join(parts) if parts else ""

    def reset(self):
        """대화 세션 초기화"""
        self.terms.clear()
        self.topic = ""
        self.last_speaker = ""
        self.topic_keywords.clear()
        self.sentence_count = 0


# 전역 문맥 관리자
context_manager = ConversationContext(max_terms=20)

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
    calibration_chunks = 3  # 처음 3개 청크(약 6초)로 주변 소음 측정

    while is_running:
        try:
            chunk = audio_queue.get(timeout=0.3)
            buffer.append(chunk)

            # 충분한 오디오가 쌓이면 처리
            total_samples = sum(c.shape[0] for c in buffer)
            if total_samples >= samples_per_chunk:
                # 버퍼 합치기
                audio_data = np.concatenate(buffer, axis=0).flatten().astype(np.float32)
                buffer = []

                # 오디오 증폭 (AUDIO_GAIN 적용)
                if AUDIO_GAIN != 1.0:
                    audio_data = audio_data * AUDIO_GAIN
                    # 클리핑 방지 (-1.0 ~ 1.0 범위로 제한)
                    audio_data = np.clip(audio_data, -1.0, 1.0)

                rms = float(np.sqrt(np.mean(audio_data ** 2)))
                current_rms = rms

                # 캘리브레이션: 처음 몇 청크로 노이즈 플로어 측정
                if not calibration_done:
                    calibration_rms_list.append(rms)
                    logging.info(f"[캘리브레이션] RMS 측정 중... ({len(calibration_rms_list)}/{calibration_chunks}) RMS={rms:.6f}")
                    if len(calibration_rms_list) >= calibration_chunks:
                        noise_floor = max(np.mean(calibration_rms_list), MIN_RMS_THRESHOLD)
                        calibration_done = True
                        logging.info(f"[캘리브레이션 완료] 노이즈 플로어: {noise_floor:.6f}, 음성 임계값: {noise_floor * SILENCE_MULTIPLIER:.6f}")
                    continue

                # 적응형 노이즈 플로어 업데이트 (환경 변화 대응)
                # - 낮은 RMS일 때만 (무음/잡음) 노이즈 플로어 업데이트
                # -话音 중에는 업데이트하지 않음
                if rms < noise_floor * SILENCE_MULTIPLIER * 0.8:
                    # 환경이 더 조용해졌다고 판단, 점진적 감소
                    noise_floor = noise_floor * NOISE_FLOOR_DECAY + rms * (1 - NOISE_FLOOR_DECAY)
                    noise_floor = max(noise_floor, MIN_RMS_THRESHOLD)
                elif rms > noise_floor * 10:
                    # 갑자기 큰 소리가 발생하면 노이즈 플로어를 급격히 올리지 않음 (순간 소음 무시)
                    pass

                # 적응형 무음 감지
                threshold = max(noise_floor * SILENCE_MULTIPLIER, MIN_RMS_THRESHOLD)
                if rms < threshold:
                    logging.debug(f"[소리입력] 무음 (RMS: {rms:.6f} < 임계값: {threshold:.6f})")
                    continue
                else:
                    logging.info(f"[소리입력] 음성 감지! (RMS: {rms:.6f} > 임계값: {threshold:.6f})")

                chunk_count += 1

                # stt_queue에 넣기 (가득 차면 가장 오래된 것 버림)
                try:
                    stt_queue.put_nowait(audio_data)
                    logging.info(f"[Audio] 청크 #{chunk_count} → STT 큐 전달 (대기: {stt_queue.qsize()})")
                except queue.Full:
                    # 큐가 가득 차면 가장 오래된 것을 버리고 새 것을 넣음
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
                lang_label = {"ja": "일본어", "en": "영어"}.get(SOURCE_LANG, "자동감지")
                logging.info(f"[STT-{worker_id}] Whisper 음성인식 시작... (언어: {lang_label})")
                start_time = time.time()
                transcribe_kwargs = {
                    "beam_size": 3,
                    "vad_filter": True,
                    "vad_parameters": {
                        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                        "threshold": 0.5,  # VAD 임계값 (0~1, 높을수록 엄격)
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
# - 최근 번역 히스토리를 문맥으로 참조하여 문장 문맥 완성
# - 여러 워커가 병렬로 번역 처리 가능
# ─────────────────────────────────────────────
def _build_context_prompt(src_text: str, src_lang: str) -> str:
    """대화 맥락을 고려한 문맥 인식 번역 프롬프트를 생성합니다."""
    lang_name = {"ja": "일본어", "en": "영어"}.get(src_lang, "외국어")

    # 스레드 안전으로 히스토리 복사
    with history_lock:
        recent = history[:CONTEXT_HISTORY] if history else []

    # 문맥 관리자의 요약 정보
    context_summary = context_manager.build_context_summary()

    # 자연스러운 번역을 위한 추가 지시
    translation_rules = []

    # 1. 생략된 주어/목적어 해결
    if recent:
        translation_rules.append(
            "• 앞 문장의 주어나 목적어가 생략된 경우, 이전 문장을 참고하여 누락된 정보를补完하세요."
        )

    # 2. 시제 연속성
    if context_manager.sentence_count > 0:
        translation_rules.append(
            "• 시제가 일관되게 유지되도록 하세요.已完成的动作用'안다/받았다', 진행 중인 것은'하고 있다' 형태."
        )

    # 3. 핵심 용어 일관성
    if context_manager.terms:
        term_list = list(context_manager.terms.items())[-8:]
        translation_rules.append(
            f"• 아래 핵심 용어의 번역을 반드시 일관되게 적용하세요: {', '.join(f'{k}({v})' for k, v in term_list)}"
        )

    # 4. 존댓말/반말 톤
    translation_rules.append(
        "• 회의/통역 상황에서는 기본적으로 존댓말(~합니다, ~니다)을 사용하되, 친근한 분위기에서는 반말도 허용."
    )

    rules_text = "\n".join(translation_rules) if translation_rules else ""

    if recent:
        # 시간순 정렬 (오래된 → 최신)
        recent_ordered = list(reversed(recent))
        context_block = "\n".join(
            f"  [{item['time']}] {item['src']} → {item['kr']}"
            for item in recent_ordered
        )
        prompt = (
            f"당신은 전문 실시간 동시통역사입니다. {lang_name}를 자연스러운 한국어로 번역해주세요.\n\n"
            f"[번역 규칙]\n{rules_text}\n\n"
            f"[핵심 용어 데이터]\n{context_summary}\n\n"
            f"[최근 대화 맥락]\n{context_block}\n\n"
            f"위 대화의 흐름을 파악하고, 아래 문장을 자연스럽게 번역하세요.\n"
            f"단어별로 따로 떼어 번역하지 말고, 전체 문장의 의미와 흐름을 고려하세요.\n"
            f"번역은 결과만 출력하세요 (설명/주석 없음):\n\n"
            f"{src_text}"
        )
    else:
        prompt = (
            f"당신은 전문 동시통역사입니다. {lang_name}를 자연스러운 한국어로 번역해주세요.\n\n"
            f"[번역 규칙]\n{rules_text}\n\n"
            f"[핵심 용어 데이터]\n{context_summary}\n\n"
            f"상황에 맞는 자연스러운 번역을 하고, 결과만 출력하세요:\n\n{src_text}"
        )

    return prompt


def translate_with_minimax(prompt: str) -> str:
    """Minimax API를 사용하여 번역 수행 (스트리밍)"""
    import os
    api_key = MINIMAX_API_KEY or os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("Minimax API 키가 설정되지 않았습니다. MINIMAX_API_KEY 환경변수를 설정하거나 translator.py에서 지정하세요.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MINIMAX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.3,
    }

    kr_text = ""
    current_kr = ""
    # OpenAI 호환 엔드포인트 사용
    with httpx.stream("POST", f"{MINIMAX_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60.0) as resp:
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


def translate_worker(ollama: "OllamaClient | None", worker_id: int):
    global current_kr, status_msg, translate_pending, error_msg

    while is_running:
        try:
            item = translate_queue.get(timeout=0.5)
            src_text = item["text"]
            src_lang = item.get("lang", SOURCE_LANG)
            translate_pending = translate_queue.qsize()
            status_msg = "🔄  번역 중..."

            try:
                prompt = _build_context_prompt(src_text, src_lang)
                logging.info(f"[번역-{worker_id}] 요청 중... (백엔드: {'Minimax' if USE_MINIMAX else 'Ollama'}, 언어: {src_lang})")
                logging.debug(f"[번역-{worker_id}] 프롬프트:\n{prompt}")
                start_time = time.time()

                # 스트리밍 응답으로 실시간 번역 결과 표시
                kr_text = ""
                current_kr = ""  # 초기화

                if USE_MINIMAX:
                    for chunk in translate_with_minimax(prompt):
                        kr_text += chunk
                        current_kr = kr_text
                else:
                    for chunk in ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        options={"temperature": 0.3},
                    ):
                        if chunk["message"]["content"]:
                            kr_text += chunk["message"]["content"]
                            current_kr = kr_text  # 실시간 업데이트 → UI에 즉시 반영

                elapsed = time.time() - start_time
                logging.info(f"[번역-{worker_id}] 완료 ({elapsed:.1f}초): {kr_text}")
            except Exception as e:
                error_msg = f"번역 오류: {e}"
                logging.error(f"[번역-{worker_id} 오류] {e}")
                status_msg = "대기 중"
                continue

            current_kr = kr_text
            error_msg = ""

            # 히스토리 추가 (스레드 안전)
            lang_flag = {"ja": "🇯🇵", "en": "🇺🇸"}.get(src_lang, "🌐")
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

            # 문맥 관리자에 새 번역 추가 (용어/주제 추적)
            context_manager.add_translation(src_text, kr_text, src_lang)

            status_msg = "✅  번역 완료"
            time.sleep(0.3)
            status_msg = "대기 중"

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
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="current", size=8),
        Layout(name="history"),
        Layout(name="footer", size=3),
    )

    # 헤더 - 소스 언어 표시
    lang_display = {"ja": "JP", "en": "EN", "auto": "AUTO"}.get(SOURCE_LANG, "?")
    lang_flag = {"ja": "🎌", "en": "🇺🇸", "auto": "🌐"}.get(SOURCE_LANG, "🌐")
    backend_name = "Minimax API" if USE_MINIMAX else "Ollama (Local)"
    header_text = Text()
    header_text.append(f"{lang_flag} ", style="bold red")
    header_text.append(f"{lang_display} → KR ", style="bold white")
    header_text.append("동시통역 ", style="bold yellow")
    header_text.append(f"| {backend_name}", style="dim")
    header_text.append(f" | 비동기 파이프라인 (STT×{STT_WORKERS} / 번역×{TRANSLATE_WORKERS})", style="dim")
    header_text.append(f" | 문맥참조: 최근 {CONTEXT_HISTORY}줄", style="dim cyan")
    layout["header"].update(
        Panel(Align.center(header_text), border_style="bright_blue", padding=(0, 2))
    )

    # 현재 번역 결과
    src_title_map = {"ja": "[bold red]🇯🇵 日本語[/bold red]", "en": "[bold green]🇺🇸 English[/bold green]", "auto": "[bold yellow]🌐 원문 (자동감지)[/bold yellow]"}
    src_style_map = {"ja": "red", "en": "green", "auto": "yellow"}
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
            src_style = "red" if lang == "ja" else ("green" if lang == "en" else "white")
            hist_table.add_row(
                item["time"],
                flag,
                Text(item["src"], style=src_style),
                item["kr"],
            )

    layout["history"].update(
        Panel(
            hist_table if history else Align.center(Text("[dim]번역 히스토리가 없습니다[/dim]")),
            title="[dim]📜 번역 히스토리 (문맥 참조)[/dim]",
            border_style="dim",
        )
    )

    # 푸터 / 상태바
    status_style = "green" if "완료" in status_msg else ("yellow" if "중" in status_msg else "dim")
    threshold = max(noise_floor * SILENCE_MULTIPLIER, MIN_RMS_THRESHOLD)
    rms_style = "green" if current_rms > threshold else "dim"
    status_line = Text()
    status_line.append(f" {status_msg}", style=status_style)
    status_line.append(f"  |  🔊 RMS: {current_rms:.4f}", style=rms_style)
    status_line.append(f" (임계: {threshold:.4f})", style="dim")
    status_line.append(f"  |  청크: {chunk_count}", style="dim")
    status_line.append(f"  |  STT: {stt_pending}", style="cyan")
    status_line.append(f"  |  번역: {translate_pending}", style="magenta")
    if error_msg:
        status_line.append(f"  ⚠️  {error_msg}", style="bold red")
    status_line.append("  |  종료: [bold]Ctrl+C[/bold]", style="dim")

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
    console.print("  [bold]3[/bold]  🌐  자동감지 (AUTO → KR)")
    console.print()
    choice = Prompt.ask(
        "[cyan]번호를 입력하세요[/cyan]",
        default="1",
        choices=["1", "2", "3"],
    )
    lang_map = {"1": "ja", "2": "en", "3": "auto"}
    return lang_map[choice]


def select_translation_backend() -> bool:
    """번역 백엔드를 선택합니다. True=Minimax, False=Ollama"""
    console.print("\n[cyan]번역 백엔드를 선택하세요:[/cyan]")
    console.print("  [bold]1[/bold]  🦙  Ollama (로컬 LLM, 무료, GPU 권장)")
    console.print("  [bold]2[/bold]  🤖  Minimax API (클라우드, API 키 필요)")
    console.print()
    choice = Prompt.ask(
        "[cyan]번호를 입력하세요[/cyan]",
        default="1",
        choices=["1", "2"],
    )
    return choice == "2"


def ask_reset_context() -> bool:
    """대화 문맥 초기화 여부를 묻습니다."""
    console.print("\n[cyan]문맥 관리 옵션:[/cyan]")
    console.print("  [bold]1[/bold]  🔄  새로운 대화 시작 (문맥 초기화)")
    console.print("  [bold]2[/bold]  📚  이전 대화 이어서 (문맥 유지)")
    console.print()
    choice = Prompt.ask(
        "[cyan]선택하세요[/cyan]",
        default="2",
        choices=["1", "2"],
    )
    return choice == "1"


def main():
    global is_running, SOURCE_LANG, USE_MINIMAX

    console.print(
        Panel(
            "[bold cyan]JP / EN → KR 동시통역기[/bold cyan]\n"
            "[dim]로컬 Whisper + Ollama/Minimax 기반 실시간 번역 (비동기 파이프라인)[/dim]\n"
            "[dim]✨ 최근 히스토리 문맥 참조 번역 지원[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # 1. 번역 백엔드 선택
    console.print("\n[cyan]1/5  번역 백엔드 선택[/cyan]")
    USE_MINIMAX = select_translation_backend()
    if USE_MINIMAX:
        console.print("[yellow]⚠️  Minimax API 사용 모드[/yellow]")
        console.print("[dim]  MINIMAX_API_KEY 환경변수를 설정하세요.[/dim]")
    else:
        console.print("[green]✓ Ollama 로컬 LLM 사용 모드[/green]")

    # 2. Ollama 연결 확인 (Ollama 모드일 때만)
    ollama = None
    if not USE_MINIMAX:
        console.print("\n[cyan]2/5  Ollama 연결 확인 중...[/cyan]")
        try:
            ollama = OllamaClient(host=OLLAMA_HOST)
            models = ollama.list()
            model_names = [m.model for m in models.models]
            console.print(f"[green]✓ 사용 가능한 모델: {', '.join(model_names)}[/green]")

            # 모델 선택 메뉴
            preset_models = ["gemma4:e4b", "aya-expanse:8b", "llama3.2:3b", "qwen2.5:3b"]
            available_presets = [m for m in preset_models if any(m.split(":")[0] in n for n in model_names)]
            if available_presets:
                console.print("\n[cyan]사용 가능한 로컬 모델:[/cyan]")
                for i, m in enumerate(available_presets, 1):
                    default_marker = " (기본)" if m == OLLAMA_MODEL else ""
                    console.print(f"  [bold]{i}[/bold]  {m}{default_marker}")

            if OLLAMA_MODEL not in model_names and not any(OLLAMA_MODEL.split(":")[0] in m for m in model_names):
                console.print(
                    f"[yellow]⚠️  기본 모델 '{OLLAMA_MODEL}'을 찾을 수 없습니다.[/yellow]\n"
                    f"[dim]  → 터미널에서: [bold]ollama pull {OLLAMA_MODEL}[/bold][/dim]"
                )
                if available_presets:
                    choice = Prompt.ask(
                        "[cyan]사용할 모델을 선택하세요[/cyan]",
                        default="1",
                    )
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(available_presets):
                            globals()["OLLAMA_MODEL"] = available_presets[idx]
                    except ValueError:
                        pass
                else:
                    alt = Prompt.ask(
                        f"[cyan]사용할 모델명을 입력하세요[/cyan]",
                        default=model_names[0] if model_names else OLLAMA_MODEL,
                    )
                    globals()["OLLAMA_MODEL"] = alt
            console.print(f"[green]✓ 선택된 모델: {OLLAMA_MODEL}[/green]")
        except Exception as e:
            console.print(
                f"[red]✗ Ollama 연결 실패: {e}[/red]\n"
                "[dim]Ollama가 실행 중인지 확인하세요: [bold]ollama serve[/bold][/dim]"
            )
            sys.exit(1)
    else:
        # Minimax 모드: API 키 확인
        import os
        if not MINIMAX_API_KEY and not os.environ.get("MINIMAX_API_KEY"):
            console.print(
                "[yellow]⚠️  MINIMAX_API_KEY 환경변수가 설정되지 않았습니다.[/yellow]\n"
                "[dim]  번역이 실패할 수 있습니다. 계속하려면 Enter를 누르세요.[/dim]"
            )
            Prompt.ask("[cyan]계속[/cyan]", default="")

    # 3. Whisper 모델 로드
    console.print(f"\n[cyan]{'3/5' if not USE_MINIMAX else '2/5'}  Whisper 모델 로드 중 ({WHISPER_MODEL})...[/cyan]")
    try:
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        console.print("[green]✓ Whisper 모델 로드 완료[/green]")
    except Exception as e:
        console.print(f"[red]✗ Whisper 로드 실패: {e}[/red]")
        sys.exit(1)

    # 4. 소스 언어 선택
    SOURCE_LANG = select_language()
    lang_display = {"ja": "일본어 (JP)", "en": "영어 (EN)", "auto": "자동감지 (AUTO)"}.get(SOURCE_LANG, "?")
    console.print(f"[green]✓ 소스 언어: {lang_display}[/green]")

    # 4-1. 문맥 초기화 여부
    if ask_reset_context():
        context_manager.reset()
        console.print("[yellow]🔄 문맥이 초기화되었습니다. 새로운 대화를 시작합니다.[/yellow]")
    else:
        console.print("[dim]📚 이전 대화 문맥을 유지합니다.[/dim]")

    # 5. 오디오 장치 선택
    step = "4/5" if USE_MINIMAX else "5/5"
    console.print(f"\n[cyan]{step}  오디오 장치 선택[/cyan]")
    device_idx = select_device()
    dev_info = sd.query_devices(device_idx)
    device_name = dev_info['name']
    console.print(f"[green]✓ 선택된 장치: {device_name}[/green]\n")
    logging.info(f"--- 프로그램 시작 (비동기 파이프라인) ---")
    logging.info(f"번역 백엔드: {'Minimax API' if USE_MINIMAX else 'Ollama'}")
    logging.info(f"소스 언어: {lang_display}")
    logging.info(f"선택된 오디오 장치: [{device_idx}] {device_name}")
    logging.info(f"STT 워커: {STT_WORKERS}개, 번역 워커: {TRANSLATE_WORKERS}개")
    logging.info(f"문맥 참조 히스토리: 최근 {CONTEXT_HISTORY}줄")

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

    # 3단계: 번역 워커 (병렬)
    translate_threads = []
    for i in range(TRANSLATE_WORKERS):
        t = threading.Thread(
            target=translate_worker,
            args=(ollama, i),
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

    # 오디오 스트림 + Live UI
    try:
        with sd.InputStream(
            device=device_idx,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms 블록
        ):
            with Live(build_ui(), refresh_per_second=4, screen=True, console=console) as live:
                while is_running:
                    live.update(build_ui())
                    time.sleep(0.25)
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
