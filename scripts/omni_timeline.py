#!/usr/bin/env python
"""Per-turn latency timeline on the PROTO path (omni mode), relative to the end
of speech. Runs scripts/omni_probe.py --via-viewer on each wav, then reads the
wall-clock stamps the server logs (2026-11-04 instrumentation):

  [spec] started on the pause (...) @T       speculation start (the pause)
  [timing] llm first token +N ms ... @T      Thinker's first token
  [timing] first sentence to TTS @T          first sentence handed to TTS
  [timing] first PCM +N ms ... @T            its first PCM
  [TTFA] ... wall endpoint@T first_pcm@T     VAD endpoint / turn's first voice
  [render] direct PCM: first chunk ... @T    first audio sent to the viewer

and prints them as ms after the last speech frame. The released speculation
is the last one started before the endpoint.

    .venv/bin/python scripts/omni_timeline.py --cam pic.jpg a.wav b.wav [--runs 2]
Probe turns land in the render tab's history: truncate it afterwards.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "logs", "server.log")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"@(\d\d):(\d\d):(\d\d)\.(\d\d\d)")
KEYS = [
    ("pause", "[spec] started on the pause"),
    ("llm_req", "[timing] llm request sent"),
    ("llm_tok", "[timing] llm first token"),
    ("sent", "[timing] first sentence to TTS"),
    ("tts_req", "[timing] tts request sent"),
    ("pcm", "[timing] first PCM"),
    ("viewer", "[render] direct PCM: first chunk"),
]


def wall_of(line: str, day0: float, which: int = 0) -> float | None:
    ms = STAMP.findall(line)
    if len(ms) <= which:
        return None
    hh, mm, ss, mmm = (int(v) for v in ms[which])
    return day0 + hh * 3600 + mm * 60 + ss + mmm / 1000


def events(since: float) -> list[tuple[float, str, float | None]]:
    day = time.strftime("%Y-%m-%d", time.localtime(since))
    day0 = time.mktime(time.strptime(day, "%Y-%m-%d"))
    out = []
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            if not line.startswith(day):
                continue
            if "[TTFA]" in line and "endpoint@" in line:
                m = re.search(r"endpoint@(\d\d):(\d\d):(\d\d)\.(\d\d\d)", line)
                if m:
                    hh, mm, ss, mmm = (int(v) for v in m.groups())
                    t = day0 + hh * 3600 + mm * 60 + ss + mmm / 1000
                    if t >= since - 5:
                        out.append((t, "endpoint", None))
                continue
            for key, needle in KEYS:
                if needle in line:
                    t = wall_of(line, day0)
                    if t is not None and t >= since - 5:
                        out.append((t, key, None))
                    break
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--cam", default="")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    ends = []
    for _ in range(args.runs):
        for w in args.wavs:
            cmd = [sys.executable, os.path.join(HERE, "omni_probe.py"), w, "--via-viewer", "--window", "9"]
            if args.cam:
                cmd += ["--cam", args.cam]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
            m = re.search(r"speech_end_wall=([0-9.]+)", out)
            if m:
                ends.append((os.path.basename(w), float(m.group(1))))
    time.sleep(1.0)
    ev = events(t0)
    cols = ["pause", "llm_req", "llm_tok", "sent", "tts_req", "pcm", "endpoint", "viewer"]
    rows = []
    print(f"{'wav':10} " + " ".join(f"{c:>9}" for c in cols) + "   (ms after the last speech frame)")
    for i, (name, end) in enumerate(ends):
        nxt = ends[i + 1][1] if i + 1 < len(ends) else end + 12
        win = [e for e in ev if end - 1.0 <= e[0] < nxt - 1.0]
        endpoint = next((t for t, k, _ in win if k == "endpoint" and t >= end), None)
        starts = [t for t, k, _ in win if k == "pause" and (endpoint is None or t <= endpoint)]
        spec = starts[-1] if starts else None
        row = {"pause": spec, "endpoint": endpoint}
        for k in ("llm_req", "llm_tok", "sent", "tts_req", "pcm", "viewer"):
            ref = spec if spec is not None else end
            row[k] = next((t for t, kk, _ in win if kk == k and t >= ref), None)
        rel = {k: (round((v - end) * 1000) if v is not None else None) for k, v in row.items()}
        rows.append(rel)
        print(f"{name:10} " + " ".join(f"{(str(rel[c]) if rel[c] is not None else '-'):>9}" for c in cols))
    med = {c: statistics.median([r[c] for r in rows if r[c] is not None]) if any(r[c] is not None for r in rows) else None for c in cols}
    print(f"{'MEDIAN':10} " + " ".join(f"{(f'{med[c]:.0f}' if med[c] is not None else '-'):>9}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
