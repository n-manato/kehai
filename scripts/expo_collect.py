#!/usr/bin/env python3
"""Collect what happened at an exhibition, from outside the server (read-only).

    .venv/bin/python -u scripts/expo_collect.py --until 13:05 [--out logs/expo/<date>]

Writes into the out dir:
  cam/HHMMSS.jpg     the PROTO camera once a second (/render-stream/camera.jpg)
  mic.jsonl          the server VAD's level / probability / state, 2 per second
  motion.jsonl       the render tab's motion player + frame rate, every 2 s (CDP :9222)
  health.jsonl       ports, GPU memory, disk, viewer statistics, every 30 s
  addressee/, omni_audio/   copies of the rotating review folders (300 / 60 newest
                     in logs/ — copied every 30 s so the whole show is kept)
  server.log         the server log lines of the run (copied at the end)

Together with logs/asr_debug (touch logs/asr_debug/ENABLED: the viewer's mic and
every VAD segment) this is enough to replay each utterance against what the
camera saw. Faces and voices of visitors: keep it on this box, delete when done.

Exits 0 at --until. A second run with --watch records nothing and exits 3 as soon as something is wrong (a
port down, no viewer for 60 s, capture under 50 fps for 30 s, a traceback, a
reply that survived its cancel) and prints what, so whoever started it is told.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:12394"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def get(url: str, timeout: float = 3.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        return None


def port_up(url: str) -> bool:
    return get(url, 4.0) is not None


async def cam_loop(out: Path, stop: float) -> None:
    (out / "cam").mkdir(exist_ok=True)
    last = b""
    while time.time() < stop:
        t = time.time()
        jpg = await asyncio.to_thread(get, f"{BASE}/render-stream/camera.jpg")
        if jpg and jpg != last:
            (out / "cam" / (dt.datetime.now().strftime("%H%M%S") + ".jpg")).write_bytes(jpg)
            last = jpg
        await asyncio.sleep(max(0.0, 1.0 - (time.time() - t)))


async def mic_loop(out: Path, stop: float) -> None:
    with open(out / "mic.jsonl", "a") as f:
        while time.time() < stop:
            raw = await asyncio.to_thread(get, f"{BASE}/render-stream/mic-level")
            if raw:
                f.write(json.dumps({"t": round(time.time(), 2), **json.loads(raw)}) + "\n")
                f.flush()
            await asyncio.sleep(0.5)


MOTION_JS = """(() => { try {
  const m = window.__vrmMotion; const o = {};
  if (m) { for (const k of ['elapsed','_holdT','_talking','robotIdle']) { const v = m[k]; if (typeof v !== 'object') o[k] = v; }
           try { o.stats = m.statsSnapshot ? m.statsSnapshot() : null; } catch (e) { o.statsErr = String(e); } }
  o.vis = document.visibilityState; o.now = performance.now();
  return JSON.stringify(o); } catch (e) { return JSON.stringify({err: String(e)}); } })()"""


async def motion_loop(out: Path, stop: float) -> None:
    import websockets

    with open(out / "motion.jsonl", "a") as f:
        while time.time() < stop:
            try:
                pages = json.loads(get("http://127.0.0.1:9222/json/list") or b"[]")
                page = next(p for p in pages if p["type"] == "page" and "render=1" in p["url"])
                async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
                    n = 0
                    while time.time() < stop:
                        n += 1
                        await ws.send(json.dumps({"id": n, "method": "Runtime.evaluate",
                                                  "params": {"expression": MOTION_JS, "returnByValue": True}}))
                        while True:
                            m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                            if m.get("id") == n:
                                break
                        v = m.get("result", {}).get("result", {}).get("value")
                        f.write(json.dumps({"t": round(time.time(), 2), "v": json.loads(v) if v else None}) + "\n")
                        f.flush()
                        await asyncio.sleep(2.0)
            except Exception as e:  # noqa: BLE001 — the tab may reload; try again
                f.write(json.dumps({"t": round(time.time(), 2), "err": repr(e)[:200]}) + "\n")
                f.flush()
                await asyncio.sleep(5.0)


def copy_new(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(exist_ok=True)
    have = {p.name for p in dst.iterdir()}
    for p in src.iterdir():
        if p.is_file() and p.name not in have:
            try:
                shutil.copy2(p, dst / p.name)
            except OSError:
                pass


class LogTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pos = path.stat().st_size if path.exists() else 0

    def new_lines(self) -> list[str]:
        try:
            size = self.path.stat().st_size
            if size < self.pos:  # rotated / restarted
                self.pos = 0
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        return [ANSI.sub("", ln) for ln in data.decode("utf-8", "replace").splitlines()]


async def health_loop(out: Path, stop: float, watch: bool, state: dict) -> None:
    tail = LogTail(ROOT / "logs" / "server.log")
    no_viewer_since = low_fps_since = None
    stat_re = re.compile(r"made ([0-9.]+) fps, sent ([0-9.]+), dropped (\d+) \| viewer: in (\d+) drawn (\d+) fps")
    with open(out / "health.jsonl", "a") as f:
        while time.time() < stop:
            now = time.time()
            ports = {
                "kehai": port_up(f"{BASE}/render-stream/mic-level"),
                "thinker": port_up("http://127.0.0.1:8808/v1/models"),
                "tts": port_up("http://127.0.0.1:8809/v1/models"),
                "gateway": port_up("http://127.0.0.1:3000/health") or port_up("http://127.0.0.1:3000/"),
                "render_tab": port_up("http://127.0.0.1:9222/json/version"),
            }
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            free_gb = shutil.disk_usage(ROOT).free / 1e9
            lines = tail.new_lines()
            faults = []
            stats = None
            for ln in lines:
                m = stat_re.search(ln)
                if m:
                    stats = [float(x) for x in m.groups()]
                if "Traceback (most recent call last)" in ln or "survived its cancel" in ln or "| ERROR" in ln:
                    faults.append(ln[:240])
                if "viewer disconnected" in ln:
                    state["viewer"] = False
                if "viewer connected" in ln:
                    state["viewer"] = True
            if stats is None:
                no_viewer_since = no_viewer_since or now  # no statistics line = nothing is being captured
            else:
                no_viewer_since = None
                low_fps_since = (low_fps_since or now) if (stats[0] < 50 or stats[4] < 45) else None
            f.write(json.dumps({"t": round(now, 1), "ports": ports, "gpu": gpu, "free_gb": round(free_gb, 1),
                                "render": stats, "faults": faults}) + "\n")
            f.flush()
            if not watch:
                copy_new(ROOT / "logs" / "addressee", out / "addressee")
                copy_new(ROOT / "logs" / "omni_audio", out / "omni_audio")
            if watch:
                why = []
                why += [f"{k} is down" for k, v in ports.items() if not v and k != "gateway"]
                if no_viewer_since and now - no_viewer_since > 60:
                    why.append("no viewer statistics for 60 s (PROTO disconnected, or capture stopped)")
                if low_fps_since and now - low_fps_since > 30:
                    why.append(f"low frame rate for 30 s: {stats}")
                why += faults[:3]
                if free_gb < 5:
                    why.append(f"disk nearly full: {free_gb:.1f} GB free")
                if why:
                    state["fault"] = why
                    return
            await asyncio.sleep(30.0)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="HH:MM today")
    ap.add_argument("--out", default="")
    ap.add_argument("--watch", action="store_true", help="watch only: record nothing, exit 3 on a fault")
    a = ap.parse_args()
    h, m = map(int, a.until.split(":"))
    stop = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()
    out = Path(a.out) if a.out else ROOT / "logs" / "expo" / dt.date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    if a.watch:
        # A second run beside the recording one: it writes its own health file and leaves
        # on the first fault, so the recorder itself never stops for one.
        out = out / "watch"
        out.mkdir(exist_ok=True)
        await health_loop(out, stop, True, state)
        if state.get("fault"):
            print(dt.datetime.now().strftime("%H:%M:%S"), "FAULT:", *state["fault"], sep="\n  ")
            return 3
        return 0
    tasks = [asyncio.create_task(c) for c in (cam_loop(out, stop), mic_loop(out, stop), motion_loop(out, stop))]
    await health_loop(out, stop, False, state)
    for t in tasks:
        t.cancel()
    day = dt.date.today().isoformat()
    try:
        with open(ROOT / "logs" / "server.log", "rb") as src, open(out / "server.log", "w") as dst:
            for raw in src:
                ln = ANSI.sub("", raw.decode("utf-8", "replace"))
                if ln.startswith(day) or not ln[:4].isdigit():
                    dst.write(ln)
    except OSError as e:
        print("server.log copy failed:", e)
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
