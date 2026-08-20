"""One-command end-to-end check of the running KEHAI server (PC-B).

    .venv/bin/python scripts/e2e_probe.py ["prompt text"] [window_secs]

Connects the way the browser does, sends ONE text turn, and reports what came
back with timings. Exists because this box has no sound card and no browser
session: it is the only way to answer "is she actually speaking and moving?"
from a shell.

What it proves, in order of the pipeline:
    WS handshake -> LLM -> streaming TTS PCM -> per-sentence motion clip

Motion is checked by FETCHING the clip the way the frontend does, not just by
noting that an id was issued: an id is handed out before generation finishes,
so an id alone is compatible with a body that never moves. The fetch starts the
moment the id arrives — polling after the turn ends gives 202/404 even on a
healthy server, because clips are consumed and evicted.

Exit code 0 = speech AND motion. Anything else is a real regression.
"""

import asyncio
import json
import sys
import time
import urllib.request
from collections import Counter

import websockets

HOST = "127.0.0.1:12394"
URL = f"ws://{HOST}/client-ws"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "Say a short greeting."
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0


async def fetch_clip(mid: str, results: dict) -> None:
    """Poll /motion/clip-smpl like the frontend: 202 until generation lands."""
    loop = asyncio.get_running_loop()
    url = f"http://{HOST}/motion/clip-smpl/{mid}"
    for _ in range(40):  # 20s
        try:
            def _get():
                with urllib.request.urlopen(url, timeout=10) as r:
                    return r.status, r.read()

            status, body = await loop.run_in_executor(None, _get)
            if status == 200:
                data = json.loads(body)
                frames = data.get("joints") or data.get("frames") or []
                results[mid] = f"200, {len(frames)} frames"
                return
        except Exception as e:  # noqa: BLE001 — 202 arrives here as HTTPError too
            if "202" not in str(e):
                results[mid] = f"ERROR {e}"
                return
        await asyncio.sleep(0.5)
    results[mid] = "still 202 after 20s"


async def main() -> int:
    kinds: Counter[str] = Counter()
    first: dict[str, float] = {}
    audio_bytes = 0
    said: list[str] = []
    clips: dict[str, str] = {}
    pollers: list[asyncio.Task] = []

    async with websockets.connect(URL, max_size=None) as ws:
        try:  # drain the opening burst (set-model, config, history, ...)
            while True:
                await asyncio.wait_for(ws.recv(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        t0 = time.monotonic()
        await ws.send(json.dumps({"type": "text-input", "text": PROMPT}))
        print(f">>> {PROMPT!r}\n")

        while time.monotonic() - t0 < WINDOW:
            try:
                left = WINDOW - (time.monotonic() - t0)
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
            except asyncio.TimeoutError:
                break
            t, dt = m.get("type", "?"), time.monotonic() - t0
            kinds[t] += 1
            first.setdefault(t, dt)

            if t in ("audio", "audio-stream-chunk"):
                audio_bytes += len(m.get("audio") or m.get("data") or "")
            if t in ("audio", "audio-stream-start"):
                if (m.get("display_text") or {}).get("text"):
                    said.append(m["display_text"]["text"])
                mid = ((m.get("actions") or {}).get("motion") or {}).get("id")
                if mid and mid not in clips:
                    clips[mid] = "pending"
                    pollers.append(asyncio.create_task(fetch_clip(mid, clips)))
            if t == "control" and m.get("text") == "conversation-chain-end":
                break

        if pollers:
            await asyncio.gather(*pollers, return_exceptions=True)

    print("--- message classes ---")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<24} x{v:<4} first@{first.get(k, float('nan')):.2f}s")
    ttfa = first.get("audio-stream-chunk", first.get("audio", float("nan")))
    print(f"\n  time to first audio  {ttfa:.2f}s")
    print(f"  audio payload        {audio_bytes / 1024:.0f} KB")
    print(f"  spoken               {' '.join(said)[:160] or 'NONE'}")
    print("  motion clips:")
    for mid, state in clips.items():
        print(f"    {mid[:8]} -> {state}")

    spoke = audio_bytes > 0
    moved = any("frames" in s and not s.startswith("200, 0") for s in clips.values())
    print("\nRESULT:", "PASS" if spoke and moved else f"FAIL (speech={spoke} motion={moved})")
    return 0 if spoke and moved else 1


sys.exit(asyncio.run(main()))
