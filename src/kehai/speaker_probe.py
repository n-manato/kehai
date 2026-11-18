"""Whose voice is this turn? Records only — nothing decides on it yet.

Why (2026-11-09 04:02): with someone standing in front of her, every sound the
microphone hears is taken as theirs. The TV beside the PROTO ("Welcome to 20
Minutes of Telltale...") was joined onto the visitor's question and answered.
The PROTO, its microphone and the TV cannot be changed, so the one thing left
to tell them apart by is the voice itself: the visitor's voice, heard on their
first turn, against each later turn.

On 11-07 the same idea failed on the room's recordings (TV and room voices came
out close to the visitor's, short turns scattered). Before anything acts on it,
this keeps — for every turn of the render tab — two public speaker embeddings
(WeSpeaker ResNet34 and 3D-Speaker CAM++, VoxCeleb, via sherpa-onnx) in
logs/speaker_probe.jsonl, with the time, so they can be matched against the
transcripts and decisions in the server log and the separation measured on real
visitor and TV turns. Embeddings only: no audio is kept.

SPEAKER_PROBE=0 turns it off.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from loguru import logger

ENABLED = os.getenv("SPEAKER_PROBE", "1") == "1"
_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "ws": _ROOT / "models/speaker/wespeaker_en_voxceleb_resnet34_LM.onnx",
    "cp": _ROOT / "models/speaker/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
}
OUT = _ROOT / "logs/speaker_probe.jsonl"

_lock = threading.Lock()
_ext: dict | None = None
_recent: deque = deque(maxlen=6)  # (t, {name: emb}) of the last turns, for the live log line


def _load() -> dict:
    global _ext
    if _ext is None:
        import sherpa_onnx

        _ext = {}
        for k, p in MODELS.items():
            if p.exists():
                cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(p), num_threads=2)
                _ext[k] = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        logger.info(f"[speaker] probe loaded {sorted(_ext)}")
    return _ext


def _embed(pcm: np.ndarray) -> dict:
    out = {}
    for k, ex in _load().items():
        s = ex.create_stream()
        s.accept_waveform(16000, pcm)
        s.input_finished()
        e = np.asarray(ex.compute(s), np.float32)
        out[k] = e / (np.linalg.norm(e) + 1e-9)
    return out


def _run(wav_b64: str, t_end: float) -> None:
    try:
        raw = base64.b64decode(wav_b64)
        pcm = np.frombuffer(raw[44:], np.int16).astype(np.float32) / 32768.0
        if len(pcm) < 0.4 * 16000:
            return
        with _lock:
            emb = _embed(pcm)
            sims = []
            for t0, e0 in _recent:
                sims.append(f"{t_end - t0:.0f}s ago " + "/".join(f"{float(emb[k] @ e0[k]):.2f}" for k in emb if k in e0))
            _recent.append((t_end, emb))
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "wall": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "mono": round(t_end, 3),
                    "dur": round(len(pcm) / 16000, 2),
                    "rms_db": round(20 * np.log10(np.sqrt(np.mean(pcm ** 2)) + 1e-9), 1),
                    **{k: [round(float(x), 4) for x in v] for k, v in emb.items()},
                }) + "\n")
        logger.info(f"[speaker] {len(pcm) / 16000:.1f}s turn, ws/cp similarity to the last turns: {'; '.join(sims[-3:]) or '-'}")
    except Exception as e:  # noqa: BLE001 — a probe never breaks a turn
        logger.debug(f"[speaker] probe failed: {e!r}")


def record(wav_b64: str) -> None:
    """Fire and forget: embed this turn's audio (16 kHz WAV, base64) off the loop."""
    if ENABLED and wav_b64:
        threading.Thread(target=_run, args=(wav_b64, time.monotonic()), daemon=True).start()
