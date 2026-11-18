"""Is the face in front moving its mouth while the voice is heard? Picture only.

Why (2026-11-09): the audio-visual lip model (active_speaker.py, LR-ASD) scored
the visitor talking straight at her on the PROTO as silent, at every audio/video
offset tried — the PROTO's sound (cut at 4 kHz, noise-suppressed) is not what it
was trained on. Google's "Look and Talk" uses lip movement as one cue among head
orientation and proximity; here it is measured from the picture alone, so the
bad sound cannot spoil it, and the sound is used only for its loudness envelope
(when someone is talking), which a 4 kHz cut keeps.

Per face track, over the utterance (crops from active_speaker.TRACKS, 112x112,
the face from the eyebrows to the chin):
  mouth / upper  — median frame-to-frame change in the mouth region against the
                   same in the eye region: the head moving moves both, talking
                   moves the mouth. Around 1 for a still or nodding face.
  corr @ lag     — correlation of the mouth's change with the voice's loudness,
                   searched over the picture arriving 0-0.8 s after the sound.

LIP=1 turns the crops and this check on. It only logs ([lip]) — nothing decides
on it until it has been checked against real visitors on the PROTO.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import active_speaker

ENABLED = os.getenv("LIP", "0") == "1"
# Regions of the 112x112 crop (rows, cols). The crop runs from ~0.3 face-heights
# above the face centre to ~1.1 below, so the eyes sit near the top and the mouth
# in the lower third (checked on PROTO crops, 2026-11-09).
MOUTH = (slice(62, 94), slice(30, 82))
UPPER = (slice(8, 40), slice(18, 94))


def _changes(crops: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, mouth change, upper-face change) between consecutive crops."""
    ts, dm, du = [], [], []
    prev = None
    for c in crops:
        g = c[1].astype(np.float32)
        if prev is not None:
            dm.append(float(np.mean(np.abs(g[MOUTH] - prev[MOUTH]))))
            du.append(float(np.mean(np.abs(g[UPPER] - prev[UPPER]))))
            ts.append(c[0])
        prev = g
    return np.array(ts), np.array(dm), np.array(du)


# The picture statistics LR-ASD was trained on (asd_model.ASDModel.visual).
_TRAIN_MEAN, _TRAIN_STD = 0.4161 * 255, 0.1688 * 255


def _normalise(clip: np.ndarray) -> np.ndarray:
    """The clip's brightness and contrast brought to the training pictures'
    (one gain for the whole clip, so the frame-to-frame change is kept). On the
    TalkNet demo a washed-out or low-contrast picture (mean 196 / std 13) pushes
    the talking face to -0.4..+0.1; normalised it is back to +0.2..+1.9.
    Demo crops: mean 117, std 33, noise (upper-face change per frame) 5.3."""
    return np.clip((clip - clip.mean()) / (clip.std() + 1e-3) * _TRAIN_STD + _TRAIN_MEAN, 0, 255)


def _unused_envelope(audio16k: np.ndarray, t0: float, times: np.ndarray, win: float = 0.04) -> np.ndarray:
    out = np.zeros(len(times))
    h = int(win * 16000)
    for k, t in enumerate(times):
        i = int((t - t0) * 16000)
        seg = audio16k[max(0, i - h):max(0, i + h)]
        out[k] = np.sqrt(np.mean(seg ** 2)) if len(seg) else np.nan
    return out


BASE_S = 3.0  # the same face before the voice began, for comparison


LAST_BEST: dict = {}  # end_t -> (voiced score, share of frames > 0, voiced frames) of the best face


def check(t0: float, t1: float, audio16k: np.ndarray) -> str:
    """One line per near face: LR-ASD's picture-only head ("is this face
    speaking", lossV — trained beside the audio-visual head, uses no sound).

    2026-11-09, PROTO logs: the visitor talking scored -0.5..+0.4 on sentences
    (15-60 % of frames > 0), standing silent while the robot next to the PROTO
    talked -0.7..-1.4 (0-10 %) — but 1 s utterances overlapped the silent range.
    So two sharper numbers are logged, to be checked before anything decides:
      voiced  — the score over the frames where the voice is loud only (the
                window also holds the lead-in and the pauses);
      rise    — voiced minus the SAME face's score in the BASE_S before the
                voice began: a talker's rises, a listener's stays (the camera's
                noise shifts everyone's score down by the same amount).
    """
    import torch

    m = active_speaker._load()
    if m is None:
        return "no model"
    # where in t0..t1 the voice is loud: 40 ms RMS within 20 dB of its peak
    hop = 640
    n = len(audio16k) // hop
    rms = np.sqrt(np.mean(audio16k[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12) if n else np.zeros(0)
    loud = rms > rms.max() * 0.1 if n else np.zeros(0, bool)
    lines, best = [], None
    for tr, crops in active_speaker.TRACKS.snapshot(t0 - BASE_S - 0.4, t1 + 0.3):
        inside = [c for c in crops if t0 - 0.3 <= c[0] <= t1 + 0.3]
        if len(inside) < max(10, 0.4 * (t1 - t0) * 25):
            continue
        grid = np.arange(int(np.ceil(crops[0][0] * 25)), int(crops[-1][0] * 25)) / 25
        if len(grid) < 10:
            continue
        clip = active_speaker._grid(crops, grid).astype(np.float32)
        with torch.no_grad():
            lg = m.visual_logits(torch.from_numpy(active_speaker._visual(clip))).numpy()
        during = (grid >= t0) & (grid <= t1)
        k = np.clip(((grid - t0) * 16000 / hop).astype(int), 0, max(n - 1, 0))
        voiced = during & (loud[k] if n else False)
        before = (grid < t0 - 0.3) & (grid >= t0 - BASE_S)
        f = lambda mask: float(np.mean(lg[mask])) if mask.sum() >= 8 else float("nan")
        whole, v, b = f(during), f(voiced), f(before)
        rise = v - b
        pos = float(np.mean(lg[voiced] > 0)) if voiced.sum() >= 8 else float("nan")
        px = float(np.median([c[2] for c in inside]))
        facing = float(np.mean([c[3] for c in inside]))
        if np.isfinite(v) and (best is None or v > best[0]):
            best = (v, rise, pos, tr.id, int(voiced.sum()))
        lines.append(
            f"#{tr.id} whole {whole:+.2f}, voiced {v:+.2f} ({pos:.0%} > 0, {int(voiced.sum())} fr), "
            f"before {b:+.2f} ({int(before.sum())} fr), rise {rise:+.2f}, {px:.0f} px, facing {facing:.0%}"
        )
    LAST_BEST.clear()
    if best is not None:
        LAST_BEST[round(t1, 3)] = (best[0], best[2], best[4])
    if not lines:
        return "no face crops"
    head = f"best #{best[3]} voiced {best[0]:+.2f} rise {best[1]:+.2f} pos {best[2]:.0%} | " if best else ""
    return head + "; ".join(lines)


def check_b64(wav_b64: str, end_t: float) -> str:
    import base64

    pcm = np.frombuffer(base64.b64decode(wav_b64)[44:], dtype=np.int16).astype(np.float32) / 32768.0
    pcm = pcm[-int(4.0 * 16000):]
    t0 = end_t - len(pcm) / 16000
    tc = time.perf_counter()
    out = check(t0, end_t, pcm)
    return f"{out} [{(time.perf_counter() - tc) * 1000:.0f} ms]"


def best_b64(wav_b64: str, end_t: float):
    """(voiced score, share of voiced frames > 0, voiced frames) of the face that
    looks most like the talker, or None when no face was seen well enough."""
    line = check_b64(wav_b64, end_t)
    return LAST_BEST.get(round(end_t, 3)), line
