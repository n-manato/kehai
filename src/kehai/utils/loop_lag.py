"""Event-loop lag monitor (LOOP_LAG=1): a task that asks to wake every 5 ms
and records how late it actually woke. Everything in KEHAI shares one asyncio
loop — the video relay to the PROTO viewer, the server VAD on every 20 ms mic
frame, Whisper and Thinker streams — so a late wake-up here is latency added
to every stage, including the moment the Thinker's first token is read
(replayed outside KEHAI the same request took 62-91 ms to its first token,
inside 103-119 ms; 2026-11-04). Logs p50/p95/max every 10 s."""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

_TASK: asyncio.Task | None = None


_last_tick = time.monotonic()


_IDLE_FILES = {"threading.py", "queue.py", "selectors.py", "loop_lag.py"}
_IDLE_FUNCS = {"wait", "_wait_for_tstate_lock", "select", "poll", "accept", "recv", "recv_into", "sleep", "get", "_worker"}


def _watchdog(main_ident: int) -> None:
    """A plain thread: when the loop has not ticked for 100 ms, log the main
    thread's stack once per stall — the code that is blocking the loop."""
    import sys
    import threading
    import traceback

    reported = 0.0
    # LOOP_LAG_STACK_MS: the stall worth a stack. 100 by default; the PROTO viewer's
    # audio has only ~60 ms in hand (DIRECT_LEAD_S + its 20 ms jitter floor), so a
    # 40 ms threshold is what catches the stalls behind an audible gap.
    limit = float(os.getenv("LOOP_LAG_STACK_MS", "100")) / 1000
    while True:
        time.sleep(0.01)
        stalled = time.monotonic() - _last_tick
        if stalled > limit and _last_tick != reported:
            reported = _last_tick
            frame = sys._current_frames().get(main_ident)
            if frame is None:
                continue
            stack = traceback.extract_stack(frame)[-12:]
            where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(stack))
            # The main thread's own stack often just shows where it waits for the GIL
            # (an ssl read, a log write). Who HOLDS it: the other threads running Python
            # right now — idle ones (waiting on a lock, a queue, a socket) are skipped.
            busy = []
            for tid, fr in sys._current_frames().items():
                if tid in (main_ident, threading.get_ident()):
                    continue
                st = traceback.extract_stack(fr)
                top = st[-1] if st else None
                if top is None or os.path.basename(top.filename) in _IDLE_FILES or top.name in _IDLE_FUNCS:
                    continue
                busy.append(" <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(st[-5:])))
            logger.warning(
                f"[loop-lag] loop blocked {stalled * 1000:.0f} ms+ in: {where}"
                + (f" || other threads: {' | '.join(busy)}" if busy else "")
            )


async def _run() -> None:
    global _last_tick
    import threading

    threading.Thread(target=_watchdog, args=(threading.main_thread().ident,), daemon=True, name="loop-lag-watchdog").start()
    lags: list[float] = []
    last_log = time.monotonic()
    while True:
        t = time.monotonic()
        _last_tick = t
        await asyncio.sleep(0.005)
        _last_tick = time.monotonic()
        lags.append((time.monotonic() - t - 0.005) * 1000)
        if time.monotonic() - last_log >= 10:
            lags.sort()
            n = len(lags)
            logger.info(
                f"[loop-lag] p50 {lags[n // 2]:.1f} ms, p95 {lags[int(n * 0.95)]:.1f} ms, "
                f"p99 {lags[int(n * 0.99)]:.1f} ms, max {lags[-1]:.1f} ms over {n} wake-ups"
            )
            lags.clear()
            last_log = time.monotonic()


_GC_WATCH = False
_gc_t0 = 0.0


def _gc_cb(phase: str, info: dict) -> None:
    """Log a garbage collection long enough to be heard: the viewer's audio has ~60 ms in
    hand, and a full collection walks every tracked object in the process."""
    global _gc_t0
    if phase == "start":
        _gc_t0 = time.perf_counter()
        return
    dt = time.perf_counter() - _gc_t0
    if dt > 0.04:
        logger.warning(
            f"[gc] gen{info.get('generation')} collection blocked the loop {dt * 1000:.0f} ms "
            f"(collected {info.get('collected')}, uncollectable {info.get('uncollectable')})"
        )


def start() -> None:
    global _TASK, _GC_WATCH
    if not _GC_WATCH:  # always on: costs two clock reads per collection
        import gc

        gc.callbacks.append(_gc_cb)
        # Everything alive at the first connection (models, modules, config) is
        # permanent: take it out of the collector's walk. A full collection over it
        # held the loop 340 ms mid-reply (2026-11-05) — the voice to the PROTO has
        # ~60 ms in hand. Objects made later are still collected as before.
        gc.collect()
        gc.freeze()
        logger.info(f"[gc] froze {gc.get_freeze_count()} startup objects out of the collector")
        _GC_WATCH = True
    if os.getenv("LOOP_LAG", "0") != "1" or (_TASK is not None and not _TASK.done()):
        return
    try:
        _TASK = asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        pass
