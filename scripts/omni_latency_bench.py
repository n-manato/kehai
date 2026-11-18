#!/usr/bin/env python
"""End-of-speech → first audio, measured, for N runs of the probe questions
(omni mode). Runs scripts/omni_probe.py on each wav (with a camera picture so
the addressee check answers), then reads the server log for the same turns and
prints, per turn and as medians:

  speech_end->ws_audio  the probe's own clock: last speech frame sent → first
                        audio-stream-chunk received (what the client sees)
  head_start            [spec] HIT: how long the reply had been running before
                        the endpoint (≈ pause→endpoint distance)
  asr_catchup           transcript caught up N ms after the pause
  endpoint->first_pcm   [TTFA] TOTAL (0-3 ms = audio was already held)
  merged_hold           short-sentence merge wait, if any

The ear adds the render tab preroll (0.05 s), the render-stream hop and the
viewer's jitter buffer + output device (see the [render] stats line: "play
X/Y ms target ... device Z ms").

    .venv/bin/python scripts/omni_latency_bench.py --wavs a.wav b.wav --cam pic.jpg --runs 2
Then truncate the probe turns from the history (chat_history_manager.truncate_history).
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
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs", "server.log")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_probe(wav: str, cam: str, via_viewer: bool = False) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "omni_probe.py"), wav]
    if cam:
        cmd += ["--cam", cam]
    if via_viewer:
        cmd += ["--via-viewer", "--window", "9"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    m = re.search(r"first audio ([0-9.]+)s after end of speech", out)
    heard = re.search(r"heard\s*:\s*(.*)", out)
    w = re.search(r"speech_end_wall=([0-9.]+)", out)
    return {
        "ws": float(m.group(1)) if m else None,
        "heard": heard.group(1).strip() if heard else "",
        "pass": "RESULT: PASS" in out or "RESULT: SENT" in out,
        "speech_end_wall": float(w.group(1)) if w else None,
    }


def first_pcm_wall_after(since: float) -> float | None:
    """Wall time of the first [TTFA] first_pcm@ stamp logged after `since`."""
    day = time.strftime("%Y-%m-%d", time.localtime(since))
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            if "[TTFA]" not in line or not line.startswith(day):
                continue
            m = re.search(r"first_pcm@(\d\d):(\d\d):(\d\d)\.(\d\d\d)", line)
            if not m:
                continue
            hh, mm, ss, ms = (int(v) for v in m.groups())
            t = time.mktime(time.strptime(day, "%Y-%m-%d")) + hh * 3600 + mm * 60 + ss + ms / 1000
            if t > since - 0.5:
                return t
    return None


def log_turns(since: float) -> list[dict]:
    """Turn records from the log lines written after `since` (epoch seconds)."""
    recs, cur = [], {}
    day = time.strftime("%Y-%m-%d", time.localtime(since))
    with open(LOG, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            if not line.startswith(day):
                continue
            try:
                ts = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            if ts < since - 1:
                continue
            if "[spec] started on the pause" in line:
                cur = {"t": line[11:19]}
            elif "transcript caught up" in line:
                m = re.search(r"caught up (\d+) ms", line)
                cur["asr_catchup"] = int(m.group(1)) if m else None
            elif "short sentence merged" in line:
                m = re.search(r"after (\d+) ms", line)
                cur["merged_hold"] = int(m.group(1)) if m else 0
            elif "[spec] HIT" in line:
                m = re.search(r"reply started (\d+) ms before", line)
                # logged just AFTER the [TTFA] line of the same turn
                target = cur if cur else (recs[-1] if recs else cur)
                target["head_start"] = int(m.group(1)) if m else None
            elif "[TTFA]" in line:
                m = re.search(r"first_pcm=(\d+)ms", line)
                cur["endpoint_to_pcm"] = int(m.group(1)) if m else None
                recs.append(cur)
                cur = {}
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--cam", default="")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--via-viewer", action="store_true", help="the PROTO path (mic-inject 48 kHz frames); first audio = server [TTFA] wall stamp")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for r in range(args.runs):
        for w in args.wavs:
            res = run_probe(w, args.cam, args.via_viewer)
            res["wav"] = os.path.basename(w)
            if args.via_viewer and res["speech_end_wall"]:
                fp = first_pcm_wall_after(res["speech_end_wall"])
                res["ws"] = round(fp - res["speech_end_wall"], 2) if fp else None
            rows.append(res)
            print(f"  run {r + 1} {res['wav']:8} speech_end->{'server_first_pcm' if args.via_viewer else 'ws_audio'}={res['ws']!s:>6} s  {'PASS' if res['pass'] else 'FAIL'}  heard={res['heard'][:40]!r}")
    turns = log_turns(t0)
    turns = [t for t in turns if t.get("endpoint_to_pcm") is not None][-len(rows):]
    for t in turns:
        print(f"  {t.get('t')} head_start={t.get('head_start')} ms  asr_catchup={t.get('asr_catchup')} ms  endpoint->first_pcm={t.get('endpoint_to_pcm')} ms  merged_hold={t.get('merged_hold', '-')}")

    def med(vals):
        v = [x for x in vals if x is not None]
        return f"{statistics.median(v):.0f}" if v else "-"

    print(
        f"MEDIANS over {len(rows)} turns: speech_end->ws_audio={statistics.median([r['ws'] for r in rows if r['ws']]):.2f} s | "
        f"head_start={med([t.get('head_start') for t in turns])} ms | asr_catchup={med([t.get('asr_catchup') for t in turns])} ms | "
        f"endpoint->first_pcm={med([t.get('endpoint_to_pcm') for t in turns])} ms"
    )
    with open(LOG, encoding="utf-8", errors="replace") as f:
        stats = [ANSI.sub("", ln) for ln in f if "render_stream" in ln and "stats" in ln]
    if stats:
        m = re.search(r"play (\d+)/(\d+) ms target, (\d+) starves \+ device (\d+) ms", stats[-1])
        if m:
            print(f"viewer path (latest [render] stats): jitter {m.group(1)}/{m.group(2)} ms, device {m.group(4)} ms, starves {m.group(3)} | render tab preroll = localStorage.streamPreroll (0 since 2026-11-04) else 50 ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
