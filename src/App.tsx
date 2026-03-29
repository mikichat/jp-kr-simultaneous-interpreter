/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";

// Initialize Gemini
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

interface TranslationEntry {
  id: string;
  jp: string;
  kr: string;
  timestamp: Date;
  confidence?: number;
}

interface AudioDevice {
  deviceId: string;
  label: string;
}

const CHUNK_DURATION_MS = 4000; // 4초 단위로 오디오 청크 처리

export default function App() {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('default');
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentJP, setCurrentJP] = useState('');
  const [currentKR, setCurrentKR] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [history, setHistory] = useState<TranslationEntry[]>([]);
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState('');
  const [errorDetail, setErrorDetail] = useState<{ title: string; detail: string; hint: string } | null>(null);
  const [showGuide, setShowGuide] = useState(false);
  const [apiKey, setApiKey] = useState(process.env.GEMINI_API_KEY || '');
  const [showApiInput, setShowApiInput] = useState(!process.env.GEMINI_API_KEY);
  const [totalChunks, setTotalChunks] = useState(0);
  const [waveformData, setWaveformData] = useState<number[]>(Array(40).fill(0));

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const animFrameRef = useRef<number>(0);
  const isCapturingRef = useRef(false);
  const genAIRef = useRef(genAI);

  // Update genAI when apiKey changes
  useEffect(() => {
    if (apiKey) {
      genAIRef.current = new GoogleGenAI({ apiKey });
    }
  }, [apiKey]);

  // Load audio devices
  const loadDevices = useCallback(async () => {
    try {
      // Request permission first to get device labels
      await navigator.mediaDevices.getUserMedia({ audio: true }).then(s => s.getTracks().forEach(t => t.stop()));
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices
        .filter(d => d.kind === 'audioinput')
        .map(d => ({
          deviceId: d.deviceId,
          label: d.label || `마이크 (${d.deviceId.slice(0, 8)}...)`,
        }));
      setDevices(audioInputs);
      if (audioInputs.length > 0 && selectedDevice === 'default') {
        setSelectedDevice(audioInputs[0].deviceId);
      }
    } catch (err) {
      console.error('장치 목록 로드 실패:', err);
    }
  }, []);

  useEffect(() => {
    loadDevices();
  }, [loadDevices]);

  // Volume animation
  const startVolumeMonitor = useCallback((stream: MediaStream) => {
    audioContextRef.current = new AudioContext();
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 128;
    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    const waveArray = new Uint8Array(40);

    const animate = () => {
      analyserRef.current!.getByteFrequencyData(dataArray);
      const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
      setVolume(Math.min(100, (avg / 128) * 100));

      // Waveform
      analyserRef.current!.getByteTimeDomainData(waveArray);
      const wave = Array.from(waveArray).map(v => ((v - 128) / 128));
      setWaveformData(wave);

      animFrameRef.current = requestAnimationFrame(animate);
    };
    animate();
  }, []);

  const stopVolumeMonitor = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    audioContextRef.current?.close();
    audioContextRef.current = null;
    setVolume(0);
    setWaveformData(Array(40).fill(0));
  }, []);

  const translateAudioChunk = useCallback(async (audioBlob: Blob) => {
    if (!apiKey && !process.env.GEMINI_API_KEY) return;
    
    setIsTranslating(true);
    setTotalChunks(prev => prev + 1);

    try {
      const arrayBuffer = await audioBlob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      let binary = '';
      uint8Array.forEach(byte => binary += String.fromCharCode(byte));
      const base64Audio = btoa(binary);

      const mimeType = audioBlob.type || 'audio/webm';

      const response = await genAIRef.current.models.generateContent({
        model: 'gemini-2.0-flash-lite',
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType: mimeType,
                  data: base64Audio,
                },
              },
              {
                text: `이 오디오를 듣고 다음 작업을 수행해주세요:
1. 일본어 음성이 있다면 정확히 전사해주세요
2. 전사한 내용을 자연스러운 한국어로 번역해주세요

응답 형식 (JSON만):
{"jp": "일본어 원문", "kr": "한국어 번역"}

일본어 음성이 없거나 침묵이라면:
{"jp": "", "kr": ""}`,
              },
            ],
          },
        ],
      });

      const text = response.text || '';
      
      // JSON 파싱
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (parsed.jp && parsed.kr) {
          setCurrentJP(parsed.jp);
          setCurrentKR(parsed.kr);
          const entry: TranslationEntry = {
            id: Date.now().toString(),
            jp: parsed.jp,
            kr: parsed.kr,
            timestamp: new Date(),
          };
          setHistory(prev => [entry, ...prev].slice(0, 20));
        }
      }
    } catch (err: any) {
      console.error('번역 오류 전체:', err);

      let title = '번역 오류';
      let detail = '';
      let hint = '';

      // HTTP 상태 코드 파악
      const status = err?.status ?? err?.httpErrorCode ?? err?.code ?? null;
      const message = err?.message ?? String(err);
      const errBody = err?.errorDetails ?? err?.body ?? null;

      if (status === 400 || message.includes('400')) {
        title = '❌ 요청 오류 (400 Bad Request)';
        detail = `모델이 해당 요청을 처리할 수 없습니다.\n원인: ${message}`;
        hint = '오디오 포맷(webm/opus)이 지원되지 않을 수 있습니다. 잠시 후 다시 시도해주세요.';
      } else if (status === 404 || message.includes('404') || message.includes('no longer available') || message.includes('not found')) {
        title = '🚫 모델을 찾을 수 없음 (404)';
        detail = `사용 중인 AI 모델이 존재하지 않거나 지원 종료되었습니다.\n원인: ${message}`;
        hint = '앱이 최신 버전인지 확인하세요. 현재 모델(gemini-2.0-flash-lite)을 사용 중입니다.';
      } else if (status === 401 || message.includes('401') || message.includes('API_KEY_INVALID') || message.includes('UNAUTHENTICATED')) {
        title = '🔑 API 키 인증 실패 (401)';
        detail = 'Gemini API 키가 유효하지 않거나 만료되었습니다.';
        hint = 'Google AI Studio(aistudio.google.com)에서 새 API 키를 발급받아 다시 입력해주세요.';
      } else if (status === 403 || message.includes('403') || message.includes('PERMISSION_DENIED')) {
        title = '🚫 권한 없음 (403 Forbidden)';
        detail = 'API 키에 이 기능을 사용할 권한이 없습니다.';
        hint = 'API 키가 올바른 프로젝트에서 생성되었는지 확인하세요. Gemini API가 활성화되어 있어야 합니다.';
      } else if (status === 429 || message.includes('429') || message.includes('RESOURCE_EXHAUSTED') || message.includes('quota')) {
        title = '⏱️ 요청 한도 초과 (429 Rate Limit)';
        detail = 'Gemini API 무료 요청 한도를 초과했습니다.';
        hint = '잠시 후(1분 뒤) 다시 시도하거나, Google AI Studio에서 유료 플랜으로 업그레이드하세요.';
      } else if (status === 500 || message.includes('500') || message.includes('INTERNAL')) {
        title = '🔥 Gemini 서버 오류 (500)';
        detail = 'Google Gemini 서버에 일시적인 문제가 발생했습니다.';
        hint = '잠시 후 자동으로 재시도됩니다.';
      } else if (status === 503 || message.includes('503') || message.includes('unavailable')) {
        title = '🔧 서비스 일시 중단 (503)';
        detail = 'Gemini API 서버가 일시적으로 사용 불가 상태입니다.';
        hint = '수초 후 자동 재시도됩니다. 지속되면 status.cloud.google.com을 확인하세요.';
      } else if (message.includes('Failed to fetch') || message.includes('NetworkError') || message.includes('CORS')) {
        title = '🌐 네트워크 오류';
        detail = 'Gemini API 서버에 연결할 수 없습니다.';
        hint = '인터넷 연결을 확인하세요.';
      } else if (message.includes('JSON') || message.includes('parse')) {
        title = '⚠️ 응답 파싱 오류';
        detail = `AI 응답을 해석하는데 실패했습니다.\n원본 응답: ${message.slice(0, 120)}`;
        hint = '오디오가 너무 짧거나 일본어가 포함되지 않았을 수 있습니다.';
      } else {
        title = `❗ 알 수 없는 오류${status ? ` (${status})` : ''}`;
        detail = message.slice(0, 200);
        hint = '브라우저 개발자 도구(F12) 콘솔에서 자세한 내용을 확인하세요.';
      }

      if (errBody) {
        try {
          const bodyStr = typeof errBody === 'string' ? errBody : JSON.stringify(errBody, null, 2);
          detail += `\n\n서버 응답:\n${bodyStr.slice(0, 300)}`;
        } catch {}
      }

      setErrorDetail({ title, detail, hint });
      setError(title);
    } finally {
      setIsTranslating(false);
    }
  }, [apiKey]);

  const startCapture = useCallback(async () => {
    if (!apiKey && !process.env.GEMINI_API_KEY) {
      setError('Gemini API 키를 입력해주세요.');
      setShowApiInput(true);
      return;
    }

    setError('');
    
    try {
      const constraints: MediaStreamConstraints = {
        audio: selectedDevice === 'default' 
          ? true 
          : { deviceId: { exact: selectedDevice } },
        video: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      isCapturingRef.current = true;
      setIsCapturing(true);

      startVolumeMonitor(stream);

      // MediaRecorder 설정
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg',
        'audio/mp4',
      ];

      let supportedMime = 'audio/webm';
      for (const mime of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mime)) {
          supportedMime = mime;
          break;
        }
      }

      const recordChunk = () => {
        if (!isCapturingRef.current) return;

        const recorder = new MediaRecorder(streamRef.current!, {
          mimeType: supportedMime,
          audioBitsPerSecond: 16000,
        });

        chunksRef.current = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };

        recorder.onstop = async () => {
          if (chunksRef.current.length > 0) {
            const blob = new Blob(chunksRef.current, { type: supportedMime });
            if (blob.size > 1000) { // 너무 작은 청크는 무시
              await translateAudioChunk(blob);
            }
          }
          // 다음 청크 녹음 시작
          if (isCapturingRef.current) {
            setTimeout(recordChunk, 100);
          }
        };

        recorderRef.current = recorder;
        recorder.start();
        setTimeout(() => {
          if (recorder.state === 'recording') {
            recorder.stop();
          }
        }, CHUNK_DURATION_MS);
      };

      recordChunk();

    } catch (err: any) {
      console.error('캡처 오류:', err);
      if (err.name === 'NotFoundError') {
        setError('선택한 오디오 장치를 찾을 수 없습니다. 스테레오 믹스가 활성화되어 있는지 확인해주세요.');
      } else if (err.name === 'NotAllowedError') {
        setError('마이크 권한이 거부되었습니다. 브라우저 설정에서 권한을 허용해주세요.');
      } else {
        setError(`캡처 오류: ${err.message}`);
      }
      setIsCapturing(false);
      isCapturingRef.current = false;
    }
  }, [apiKey, selectedDevice, startVolumeMonitor, translateAudioChunk]);

  const stopCapture = useCallback(() => {
    isCapturingRef.current = false;
    setIsCapturing(false);

    if (recorderRef.current && recorderRef.current.state === 'recording') {
      recorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    stopVolumeMonitor();
  }, [stopVolumeMonitor]);

  const clearHistory = () => {
    setHistory([]);
    setCurrentJP('');
    setCurrentKR('');
    setTotalChunks(0);
  };

  const speakText = (text: string, lang: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    window.speechSynthesis.speak(utterance);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="app-root">
      {/* Background */}
      <div className="bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <div className="logo-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-title">JP→KR 동시통역</h1>
              <p className="logo-sub">일본어 PC 사운드 → 한국어 실시간 번역</p>
            </div>
          </div>

          <div className="header-badges">
            <span className="badge badge-ja">JA</span>
            <svg className="arrow-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="5" y1="12" x2="19" y2="12"/>
              <polyline points="12 5 19 12 12 19"/>
            </svg>
            <span className="badge badge-kr">KO</span>

            <button
              className="guide-btn"
              onClick={() => setShowGuide(true)}
              title="Windows 11 스테레오 믹스 설정 가이드"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
              </svg>
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        {/* API Key Input */}
        <AnimatePresence>
          {showApiInput && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="api-panel"
            >
              <div className="api-panel-header">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
                <span>Gemini API 키 설정</span>
              </div>
              <div className="api-input-row">
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="AIza..."
                  className="api-input"
                />
                <button
                  className="api-save-btn"
                  onClick={() => {
                    if (apiKey) setShowApiInput(false);
                  }}
                >
                  저장
                </button>
              </div>
              <p className="api-hint">
                <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
                  Google AI Studio
                </a>에서 무료로 발급받을 수 있습니다
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Device Selection */}
        <div className="section">
          <div className="section-label">
            <span className="dot" />
            오디오 입력 장치
          </div>
          <div className="device-row">
            <select
              value={selectedDevice}
              onChange={(e) => setSelectedDevice(e.target.value)}
              className="device-select"
              disabled={isCapturing}
            >
              {devices.map(d => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label}
                </option>
              ))}
            </select>
            <button className="refresh-btn" onClick={loadDevices} disabled={isCapturing} title="장치 목록 새로고침">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="23 4 23 10 17 10"/>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
            </button>
          </div>
          <p className="device-hint">
            💡 PC 사운드 캡처: <strong>스테레오 믹스</strong> 또는 <strong>WASAPI Loopback</strong> 장치를 선택하세요
          </p>
        </div>

        {/* Waveform Visualizer */}
        <div className="visualizer-container">
          <div className={`visualizer ${isCapturing ? 'is-active' : ''}`}>
            {waveformData.map((val, i) => (
              <motion.div
                key={i}
                className="bar"
                animate={{
                  height: isCapturing ? `${Math.max(4, Math.abs(val) * 60 + 4)}px` : '4px',
                  opacity: isCapturing ? 0.7 + Math.abs(val) * 0.3 : 0.3,
                }}
                transition={{ duration: 0.05 }}
              />
            ))}
          </div>

          {/* Volume meter */}
          <div className="volume-meter">
            <div className="volume-track">
              <motion.div
                className="volume-fill"
                animate={{ width: `${volume}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>
            <span className="volume-label">{Math.round(volume)}%</span>
          </div>
        </div>

        {/* Control Button */}
        <div className="control-center">
          <motion.button
            className={`capture-btn ${isCapturing ? 'is-capturing' : ''}`}
            onClick={isCapturing ? stopCapture : startCapture}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            {isCapturing ? (
              <>
                <motion.div
                  className="recording-dot"
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ repeat: Infinity, duration: 1.2 }}
                />
                캡처 중지
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                동시통역 시작
              </>
            )}
          </motion.button>

          {isCapturing && (
            <div className="status-info">
              <div className="status-dot" />
              <span>처리된 청크: {totalChunks}개</span>
              {isTranslating && <span className="translating-badge">번역 중...</span>}
            </div>
          )}
        </div>

        {/* Error Detail Panel */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.98 }}
              className="error-panel"
            >
              <div className="error-panel-header">
                <div className="error-panel-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span>{errorDetail?.title || error}</span>
                </div>
                <button
                  className="error-dismiss"
                  onClick={() => { setError(''); setErrorDetail(null); }}
                  title="닫기"
                >
                  ×
                </button>
              </div>
              {errorDetail?.detail && (
                <div className="error-detail">
                  {errorDetail.detail.split('\n').map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                </div>
              )}
              {errorDetail?.hint && (
                <div className="error-hint">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="16" x2="12" y2="12"/>
                    <line x1="12" y1="8" x2="12.01" y2="8"/>
                  </svg>
                  <span>해결 방법: {errorDetail.hint}</span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Translation Output */}
        <div className="translation-grid">
          {/* Japanese */}
          <div className="translation-card card-jp">
            <div className="card-header">
              <span className="card-lang-badge">日本語</span>
              <div className="card-actions">
                {currentJP && (
                  <>
                    <button onClick={() => speakText(currentJP, 'ja-JP')} className="card-action-btn" title="읽기">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                        <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                      </svg>
                    </button>
                    <button onClick={() => copyToClipboard(currentJP)} className="card-action-btn" title="복사">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                      </svg>
                    </button>
                  </>
                )}
              </div>
            </div>
            <div className="card-content">
              {currentJP || (
                <span className="placeholder">
                  {isCapturing ? '일본어 음성 대기 중...' : '캡처를 시작하면 여기에 일본어가 표시됩니다'}
                </span>
              )}
            </div>
          </div>

          {/* Korean */}
          <div className="translation-card card-kr">
            <div className="card-header">
              <span className="card-lang-badge">한국어</span>
              {isTranslating && (
                <motion.div
                  className="translating-bar"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="translating-progress"
                    animate={{ x: ['-100%', '100%'] }}
                    transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                  />
                </motion.div>
              )}
              <div className="card-actions">
                {currentKR && (
                  <>
                    <button onClick={() => speakText(currentKR, 'ko-KR')} className="card-action-btn" title="읽기">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                        <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                      </svg>
                    </button>
                    <button onClick={() => copyToClipboard(currentKR)} className="card-action-btn" title="복사">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                      </svg>
                    </button>
                  </>
                )}
              </div>
            </div>
            <div className="card-content card-kr-text">
              {currentKR || (
                <span className="placeholder">
                  {isCapturing ? (isTranslating ? '번역 중...' : '번역 결과 대기 중...') : '번역된 한국어가 여기에 표시됩니다'}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* History */}
        {history.length > 0 && (
          <div className="history-section">
            <div className="history-header">
              <h2 className="history-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="12 8 12 12 14 14"/>
                  <path d="M3.05 11a9 9 0 1 1 .5 4m-.5 5v-5h5"/>
                </svg>
                번역 히스토리
              </h2>
              <button className="clear-btn" onClick={clearHistory}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
                지우기
              </button>
            </div>

            <div className="history-list">
              <AnimatePresence initial={false}>
                {history.map((item) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className="history-item"
                  >
                    <div className="history-content">
                      <div className="history-jp">{item.jp}</div>
                      <div className="history-arrow">→</div>
                      <div className="history-kr">{item.kr}</div>
                    </div>
                    <div className="history-meta">
                      <span>{item.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                      <button onClick={() => speakText(item.kr, 'ko-KR')} className="history-speak-btn" title="읽기">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        </svg>
                      </button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-status">
            <div className={`status-indicator ${isCapturing ? 'is-on' : ''}`} />
            <span>{isCapturing ? '캡처 중' : '대기 중'}</span>
          </div>
          <button
            className="api-key-btn"
            onClick={() => setShowApiInput(v => !v)}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
            API 키 {apiKey ? '✓' : '설정'}
          </button>
          <span className="footer-powered">Powered by Gemini 2.0 Flash Lite</span>
        </div>
      </footer>

      {/* Guide Modal */}
      <AnimatePresence>
        {showGuide && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowGuide(false)}
          >
            <motion.div
              className="modal"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              onClick={e => e.stopPropagation()}
            >
              <div className="modal-header">
                <h3>Windows 11 스테레오 믹스 설정 가이드</h3>
                <button className="modal-close" onClick={() => setShowGuide(false)}>×</button>
              </div>
              <div className="modal-body">
                <p className="guide-intro">PC에서 재생되는 모든 소리(유튜브, 게임, 회의 등)를 캡처하려면 <strong>스테레오 믹스</strong>를 활성화해야 합니다.</p>

                <div className="guide-steps">
                  <div className="guide-step">
                    <div className="step-num">1</div>
                    <div className="step-content">
                      <strong>사운드 설정 열기</strong>
                      <p>작업 표시줄 우측 하단의 🔊 스피커 아이콘을 우클릭 → <em>사운드 설정</em></p>
                    </div>
                  </div>
                  <div className="guide-step">
                    <div className="step-num">2</div>
                    <div className="step-content">
                      <strong>고급 사운드 옵션</strong>
                      <p><em>추가 사운드 설정</em> 클릭 → 사운드 창 → <em>녹음</em> 탭 선택</p>
                    </div>
                  </div>
                  <div className="guide-step">
                    <div className="step-num">3</div>
                    <div className="step-content">
                      <strong>스테레오 믹스 활성화</strong>
                      <p>빈 곳에서 우클릭 → <em>"사용 안 함인 장치 표시"</em> 체크 → <strong>스테레오 믹스</strong> 우클릭 → <em>사용</em></p>
                    </div>
                  </div>
                  <div className="guide-step">
                    <div className="step-num">4</div>
                    <div className="step-content">
                      <strong>앱에서 장치 선택</strong>
                      <p>이 앱의 <em>오디오 입력 장치</em> 드롭다운에서 <strong>스테레오 믹스</strong>를 선택하세요</p>
                    </div>
                  </div>
                </div>

                <div className="guide-note">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span>스테레오 믹스가 없다면 VB-Audio Virtual Cable 등 가상 오디오 드라이버를 사용하세요</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
