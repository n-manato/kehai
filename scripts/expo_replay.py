#!/usr/bin/env python3
"""Play a recorded stretch of an exhibition back into the running server.

    .venv/bin/python -u scripts/expo_replay.py --dir logs/expo/2026-11-13 --from 10:00:00 --to 10:15:00

Feeds what scripts/expo_collect.py and logs/asr_debug kept — the PROTO's own
microphone (asr_debug/viewer.rec, 48 kHz int16 as the viewer sent it) and its
camera (cam/HHMMSS.jpg, one a second) — to /render-stream/mic-inject in real
time, so the server hears and sees that stretch again exactly as the viewer
would deliver it. While it runs, the real viewer's mic and camera are set aside
(the hook does that) and SHE ANSWERS OUT LOUD on the PROTO: run it with nobody
there. Afterwards it prints what the server did with the stretch, counted from
logs/server.log, so two builds / settings can be compared on the same room.

Limits: the camera was kept at 1 picture a second; each is sent ~10 times with
a changing trailing byte (so the feed does not read as frozen), i.e. people move
in 1 s steps. Her own replies differ run to run (sampling), so compare counts,
not single turns. The turns land in the render tab's history: the files created
during the run are listed at the end (delete them, then restart or reload).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import re
import struct
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
INJ = "ws://127.0.0.1:12394/render-stream/mic-inject"


def wall(day: dt.date, hms: str) -> float:
    h, m, s = map(int, hms.split(":"))
    return dt.datetime.combine(day, dt.time(h, m, s)).timestamp()


def mic_records(path: Path, t0: float, t1: float):
    """(wall time, payload) of the records between t0 and t1."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            t, n = struct.unpack("<dI", head)
            if t > t1:
                return
            if t < t0:
                f.seek(n, 1)
                continue
            yield t, f.read(n)


async def play(day_dir: Path, t0: float, t1: float) -> None:
    import websockets

    cam_dir = day_dir / "cam"
    start = time.time()
    shift = start - t0  # recorded wall time + shift = now

    async def mic(ws) -> None:
        n = 0
        for t, payload in mic_records(day_dir / "asr_debug" / "viewer.rec", t0, t1):
            delay = t + shift - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(bytes([3]) + payload)
            n += 1
        print(f"mic: {n} chunks sent")

    async def cam(ws) -> None:
        k = sent = 0
        jpg = None
        while time.time() - shift < t1:
            rec = time.time() - shift
            name = dt.datetime.fromtimestamp(rec).strftime("%H%M%S") + ".jpg"
            p = cam_dir / name
            if p.exists():
                jpg = p.read_bytes()
            if jpg:
                k += 1
                await ws.send(bytes([5]) + jpg + bytes([k % 7 + 1]))
                sent += 1
            await asyncio.sleep(0.1)
        print(f"cam: {sent} frames sent")

    async with websockets.connect(INJ, max_size=None) as a, websockets.connect(INJ, max_size=None) as b:
        await asyncio.gather(mic(a), cam(b))
    await asyncio.sleep(12)  # let the last reply finish


def report(since: float, until: float, label: str) -> None:
    day = dt.date.fromtimestamp(since).isoformat()
    a, b = (dt.datetime.fromtimestamp(x).strftime("%H:%M:%S") for x in (since, until))
    lines = []
    with open(ROOT / "logs" / "server.log", "rb") as f:
        for raw in f:
            ln = ANSI.sub("", raw.decode("utf-8", "replace")).rstrip("\n")
            if ln[:10] == day and a <= ln[11:19] <= b:
                lines.append(ln)
    c = Counter()
    keys = {
        "endpoints: to her": "[addressee] to her",
        "endpoints: not to her": "[addressee] not to her",
        "  of those: no face": "no face)",
        "  of those: over her reply, busy room": "spoken over her reply",
        "  of those: background voice": "a voice was already running",
        "  of those: no agreed words, busy room": "no words both transcripts heard",
        "replies spoken (AI response)": "AI response:",
        "replies cancelled (interrupted)": "Conversation task was successfully interrupted",
        "she stopped: 'the visitor goes on'": "the visitor goes on — she stops",
        "she stopped: barge-in confirmed": "barge-in confirmed",
        "busy room: she finishes (onset)": "onset while she speaks in a busy room",
        "busy room: speculation skipped": "the room is busy — she finishes first",
        "speculations started": "[spec] started on the pause",
        "speculations dropped: user resumed": "dropped (user resumed)",
        "fast releases": "[spec] FAST HIT",
        "retracted after release": "retracted after fast release",
        "  of those: transcripts disagree, busy room": "retracted after fast release — busy room",
        "unfinished holds": "fast release held: the words look unfinished",
        "direct PCM ran dry": "direct PCM ran dry",
        "tracebacks": "Traceback (most recent call last)",
        "survived cancel": "survived its cancel",
    }
    for ln in lines:
        for k, pat in keys.items():
            if pat in ln:
                c[k] += 1
    ttfa = sorted(int(m.group(1)) for ln in lines if (m := re.search(r"endpoint->first_pcm=(\d+)ms", ln)) and int(m.group(1)) < 60000)
    utt = sorted(float(m.group(1)) for ln in lines if (m := re.search(r"\(([0-9.]+)s utterance\)", ln)))
    print(f"\n==== {label}: server log {a}-{b} ====")
    for k in keys:
        print(f"{c[k]:6d}  {k}")
    if ttfa:
        print(f"endpoint -> first PCM: n {len(ttfa)}, median {ttfa[len(ttfa) // 2]} ms, p90 {ttfa[int(0.9 * (len(ttfa) - 1))]} ms, max {ttfa[-1]} ms")
    if utt:
        print(f"released utterances: n {len(utt)}, median {utt[len(utt) // 2]:.1f} s, max {utt[-1]:.1f} s")
    ch = sorted(int(m.group(1)) for ln in lines if (m := re.search(r"(\d+) audio chunks had gone to the viewer", ln)))
    if ch:
        print(f"audio chunks sent before a retraction: n {len(ch)}, none in {sum(c == 0 for c in ch)}, median {ch[len(ch) // 2]}, max {ch[-1]}")
    print("replies:")
    for ln in lines:
        if "AI response:" in ln:
            print("  ", ln[11:19], ln.split("AI response:", 1)[1].strip()[:110])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--from", dest="t0", required=True, help="HH:MM:SS of the recording")
    ap.add_argument("--to", dest="t1", required=True)
    ap.add_argument("--label", default="replay")
    a = ap.parse_args()
    day_dir = (ROOT / a.dir) if not Path(a.dir).is_absolute() else Path(a.dir)
    day = dt.date.fromisoformat(day_dir.name)
    t0, t1 = wall(day, a.t0), wall(day, a.t1)
    hist = ROOT / "chat_history"
    before = {p for p in hist.rglob("*.json")}
    since = time.time()
    print(f"replaying {a.t0}-{a.t1} of {day} ({(t1 - t0) / 60:.1f} min) from {dt.datetime.now():%H:%M:%S}")
    asyncio.run(play(day_dir, t0, t1))
    report(since, time.time(), a.label)
    new = sorted(p for p in hist.rglob("*.json") if p not in before)
    print("history files created during the run:", *[str(p.relative_to(ROOT)) for p in new], sep="\n  ")


if __name__ == "__main__":
    main()
