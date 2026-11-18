"""Semantic end-of-turn detection: Pipecat's Smart Turn v3.2 (BSD-2, 8 MB ONNX,
Whisper-tiny encoder + linear head) judges from the WAVEFORM whether the
speaker has finished — intonation, a trailing "and…", a half sentence — rather
than from a silence timer.

Why (2026-11-04): with the Thinker answering ~0.43 s after the speaker stops,
the fixed 0.45 s VAD endpoint became the binding constraint on most turns (the
reply's first PCM was ready at 0.40-0.43 s and waited for it), and the same
timer made her start answering inside people's mid-sentence pauses ("What time
is it? … right now in Omaha?"). The model is asked at every PAUSE (when the
speculative reply starts): COMPLETE lets the reply go out the moment its first
audio exists instead of at the endpoint; INCOMPLETE holds it past the endpoint
so a speaker who continues is merged rather than answered half-way.

Reference preprocessing (github.com/pipecat-ai/smart-turn inference.py): the
last 8 s of 16 kHz audio, zero-padded at the START to exactly 8 s, waveform
normalised to zero mean / unit variance, then WhisperFeatureExtractor
(chunk_length=8): 80 log-mel bins, n_fft 400, hop 160 -> (1, 80, 800). The
feature extractor is reimplemented here in NumPy (no transformers dependency);
checked against transformers' WhisperFeatureExtractor to < 1e-4.

Env: SMART_TURN=1 on; SMART_TURN_MODEL path (default models/smart_turn/
smart-turn-v3.2-cpu.onnx). Inference ~10-20 ms on CPU, run in a thread.
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache

import numpy as np
from loguru import logger

SR = 16000
N_SECONDS = 8
N_FFT = 400
HOP = 160
N_MELS = 80

ENABLED = os.getenv("SMART_TURN", "0") == "1"
MODEL_PATH = os.getenv("SMART_TURN_MODEL", "models/smart_turn/smart-turn-v3.2-cpu.onnx")
_NICE = int(os.getenv("SMART_TURN_NICE", "10") or 10)

_session = None
_load_failed = False


def _hz_to_mel(f):
    """Slaney mel scale (librosa htk=False), as WhisperFeatureExtractor uses."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


@lru_cache(maxsize=1)
def _mel_filters() -> np.ndarray:
    """(n_freq=201, n_mels=80) slaney-normalised triangular filters, 0-8 kHz —
    transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")."""
    n_freq = 1 + N_FFT // 2
    fft_freqs = np.linspace(0, SR // 2, n_freq)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_pts)
    fdiff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]  # (n_freq, n_mels+2)
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : N_MELS + 2] - filter_freqs[:N_MELS])
    return fb * enorm[None, :]


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    # periodic Hann, as transformers.audio_utils.window_function(400, "hann")
    return np.hanning(N_FFT + 1)[:-1]


def features(audio: np.ndarray) -> np.ndarray:
    """16 kHz float audio (any length) -> (1, 80, 800) float32 input_features."""
    x = np.asarray(audio, dtype=np.float32)
    n = N_SECONDS * SR
    if len(x) > n:
        x = x[-n:]
    elif len(x) < n:
        x = np.pad(x, (n - len(x), 0))
    x = x.astype(np.float64)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    # STFT, center=True with reflect padding
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * _window()[None, :]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2  # (frames, 201)
    mel = np.maximum(1e-10, spec @ _mel_filters())  # (frames, 80)
    log_spec = np.log10(mel).T[:, :-1]  # (80, frames-1) — Whisper drops the last frame
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[None, :, :].astype(np.float32)


def _load():
    global _session, _load_failed
    if _session is not None or _load_failed:
        return _session
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = int(os.getenv("SMART_TURN_THREADS", "1") or 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        logger.info(f"[smart-turn] loaded {MODEL_PATH}")
    except Exception as e:  # noqa: BLE001 — the timers still work without it
        _load_failed = True
        logger.warning(f"[smart-turn] unavailable ({e!r}); endpoint stays timer-only")
    return _session


def predict(audio: np.ndarray) -> float | None:
    """P(turn complete) for the utterance so far (16 kHz float). None if the
    model is unavailable. Synchronous — call through predict_async."""
    s = _load()
    if s is None or audio is None or len(audio) < SR // 4:
        return None
    out = s.run(None, {"input_features": features(audio)})
    return float(np.asarray(out[0]).reshape(-1)[0])


def _predict_low_priority(audio: np.ndarray) -> float | None:
    """predict() on a worker thread demoted to nice +10 (Linux: per-thread
    nice via its TID). At the pause the same CPU is needed by vLLM's API
    process (audio features, tokenisation) and the TTS: at normal priority
    this model slowed every later stage by 20-60 ms (2026-11-04)."""
    try:
        import threading

        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), _NICE)
    except Exception:  # noqa: BLE001 — not Linux / not permitted: run as is
        pass
    return predict(audio)


async def predict_async(audio: np.ndarray) -> tuple[float | None, float]:
    """(probability, ms) off the event loop, at low CPU priority."""
    t0 = time.perf_counter()
    p = await asyncio.to_thread(_predict_low_priority, audio)
    return p, (time.perf_counter() - t0) * 1000


def warm() -> None:
    """Load the session and run once (first ONNX run allocates)."""
    if ENABLED and _load() is not None:
        try:
            predict(np.zeros(SR, dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[smart-turn] warm-up failed: {e!r}")
