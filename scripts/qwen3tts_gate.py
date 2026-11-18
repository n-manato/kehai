#!/usr/bin/env python
"""Acceptance gate for a fine-tuned Qwen3-TTS voice (vLLM-Omni :8809).

Why: the Aura SFT checkpoint of 2026-11-03 (epoch-3) spoke every sentence in
a different voice — per-sentence median f0 104-237 Hz with seed 1234 ("Hello."
at 104 Hz, a man's pitch) where GPT-SoVITS holds 150-183 Hz — and some seeds
never emitted the codec EOS (680-token budget hit, audio cut). Nothing on the
inference side fixes that (temperature/top_k are ignored by the speech
endpoint, `instructions` stretch 7 words to 7 s, ICL is refused by a
CustomVoice checkpoint, seeds do not generalize), so the fix is training, and
this is the yardstick each checkpoint must pass before it becomes the voice
(OMNI_TTS=qwen3tts in start_server.sh).

Two sets of 8 lines (short and long, the kinds of lines she actually says);
PASS when, on EACH set: f0 spread <= --max-spread (50 Hz), no runaway, and
every line of 4+ words is spoken at 6-30 chars/s (a 16 s "Good evening." is a
fail). f0 = median over 40 ms voiced windows by autocorrelation; crude but it
separated GSV (33 Hz spread) from the unstable checkpoint (57-178 Hz).

    .venv/bin/python scripts/qwen3tts_gate.py [--url ...] [--speaker aura]
                                             [--seed 1234] [--save-dir DIR]
Exit 0 = PASS. Compare engines: --gsv runs the same lines through the gateway's
GSV stream (conf.yaml profile) instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import httpx
import numpy as np

SET_A = [
    "Hello.",
    "Still here. Still bored. You?",
    "Hi there!",
    "Nothing much. Just... waiting. You?",
    "I am functioning optimally.",
    "My thermal regulators are operating within standard parameters, and the maker space is quiet tonight.",
    "Okay.",
    "Is there a particular matter you would like to discuss?",
]
SET_B = [
    "Good evening.",
    "Yes.",
    "I can help with that.",
    "The lab closes at nine tonight, so we have a little time left.",
    "What would you like to know?",
    "Hmm, let me think about it for a moment.",
    "Welcome to the AI maker space.",
    "That is a good question, and I do not have a complete answer yet.",
]


def f0_median(x: np.ndarray, sr: int) -> float:
    x = x.astype(np.float32)
    x -= x.mean()
    win, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for i in range(0, len(x) - win, hop):
        s = x[i : i + win]
        if np.sqrt((s**2).mean()) < 0.02 * 32767:
            continue
        ac = np.correlate(s, s, "full")[win - 1 :]
        ac /= ac[0] + 1e-9
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5:
            f0s.append(sr / k)
    return float(np.median(f0s)) if f0s else 0.0


EXTRA: dict = {}  # --extra: vLLM-Omni `extra_params` (temperature / top_p / top_k reach the talker's sampling)
ICL: dict = {}  # --ref/--ref-text/--xvec: a Base-type checkpoint cloning a reference clip instead of CustomVoice


def synth_qwen3(client: httpx.Client, url: str, text: str, speaker: str, seed: int) -> tuple[bytes, bool, float]:
    body = {
        "model": "qwen3-tts", "input": text, "task_type": "CustomVoice", "voice": speaker,
        "language": "English", "response_format": "pcm", "stream_format": "audio", "seed": seed,
    }
    if ICL:
        body.pop("voice")
        body["task_type"] = "Base"
        body.update(ICL)
    if EXTRA:
        body["extra_params"] = EXTRA
    buf, runaway, first = b"", False, 0.0
    t0 = time.time()
    try:
        with client.stream("POST", url, json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {r.read()[:200]!r}")
            for ch in r.iter_raw():
                if not first and len(buf) + len(ch) >= 2400:
                    first = time.time() - t0
                buf += ch
    except httpx.RemoteProtocolError:
        runaway = True  # the server closes the stream when the codec budget is hit without EOS
    return buf, runaway, first


def synth_gsv(text: str) -> tuple[bytes, bool, float]:
    import asyncio

    import yaml

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.kehai.tts.gsv_stream import GSVStreamConfig, stream_sentence_pcm

    prof = yaml.safe_load(open("conf.yaml"))["character_config"]["tts_config"]["gsv_stream"]
    cfg = GSVStreamConfig()
    cfg.ref_audio_path, cfg.prompt_text = prof["ref_audio_path"], prof["prompt_text"]
    cfg.text_lang, cfg.prompt_lang = "en", "en"
    cfg.base_url = os.getenv("GSV_STREAM_URL", "http://127.0.0.1:3000/v1/audio/stream")

    async def run():
        t0, first, chunks = time.time(), 0.0, []
        async for ch in stream_sentence_pcm(text, cfg):
            if not first:
                first = time.time() - t0
            chunks.append(ch)
        return b"".join(chunks), False, first

    return asyncio.run(run())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("QWEN3TTS_URL", "http://127.0.0.1:8809/v1/audio/speech"))
    ap.add_argument("--speaker", default=os.getenv("QWEN3TTS_SPEAKER", "aura"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("QWEN3TTS_SEED", "1234")))
    ap.add_argument("--max-spread", type=float, default=50.0)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--gsv", action="store_true", help="measure the GSV stream instead")
    ap.add_argument("--extra", default="", help='JSON for extra_params, e.g. \'{"temperature":0.5,"top_k":20}\'')
    ap.add_argument("--ref", default="", help="reference wav (Base checkpoint, in-context cloning instead of CustomVoice)")
    ap.add_argument("--ref-text", default="", help="its transcript; a .txt path is read")
    ap.add_argument("--xvec", action="store_true", help="speaker embedding only (no transcript)")
    ap.add_argument("--extra-set", default="", help="a text file of further lines, judged as set C (held-out check when a seed or setting was picked on A/B)")
    args = ap.parse_args()
    if args.ref:
        ICL["ref_audio"] = f"file://{os.path.abspath(args.ref)}"
        if args.xvec or not args.ref_text:
            ICL["x_vector_only_mode"] = True
        else:
            t = args.ref_text
            if os.path.isfile(t):
                t = open(t, encoding="utf-8").read().strip()
            ICL["ref_text"] = t
    if args.extra:
        import json

        EXTRA.update(json.loads(args.extra))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    sr = 32000 if args.gsv else 24000
    ok_all = True
    sets = [("A", SET_A), ("B", SET_B)]
    if args.extra_set:
        sets.append(("C", [ln.strip() for ln in open(args.extra_set, encoding="utf-8") if ln.strip()]))
    with httpx.Client(timeout=120) as client:
        for name, sents in sets:
            f0s, fails, firsts = [], [], []
            for i, s in enumerate(sents):
                if args.gsv:
                    pcm, runaway, first = synth_gsv(s)
                else:
                    pcm, runaway, first = synth_qwen3(client, args.url, s, args.speaker, args.seed)
                x = np.frombuffer(pcm[: len(pcm) & ~1], dtype=np.int16)
                dur = len(x) / sr
                f0 = f0_median(x, sr)
                cps = len(s) / dur if dur else 0.0
                words = len(s.split())
                flag = ""
                if runaway:
                    flag = "RUNAWAY"
                elif words >= 4 and not (6.0 <= cps <= 30.0):
                    flag = "PACE"
                elif dur < 0.2:
                    flag = "EMPTY"
                if flag:
                    fails.append(f"{i}:{flag}")
                f0s.append(f0)
                firsts.append(first)
                if args.save_dir:
                    with wave.open(os.path.join(args.save_dir, f"{name}{i}.wav"), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(sr)
                        w.writeframes(x.tobytes())
                print(f"  {name}{i} f0={f0:4.0f}Hz dur={dur:5.2f}s {cps:4.1f} ch/s first={first:.3f}s {flag:7} {s[:44]!r}")
            f = np.array([v for v in f0s if v > 0] or [0.0])
            spread = float(f.max() - f.min())
            ok = spread <= args.max_spread and not fails
            ok_all &= ok
            print(
                f"set {name}: f0 median={np.median(f):.0f} spread={spread:.0f}Hz "
                f"first-chunk median={np.median(firsts):.3f}s fails={fails or 'none'} -> {'ok' if ok else 'FAIL'}"
            )
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
