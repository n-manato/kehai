"""Per-emotion warm pool of ready .vmd clips + on-demand generation.

The live path is `GET /pool/{emotion}/next`: it must return INSTANTLY, so a
background loop keeps `POOL_PER_EMOTION` ready clips per emotion. On-demand
`POST /motion` requests jump ahead of background refills.

Pipeline per clip:  prompt -> runner.generate (SMPL npz) -> smpl_to_mmd.retarget
-> vmd_writer.write_vmd -> cache file -> enqueue.

Concurrency: NUM_WORKERS generation slots (one per GPU). Generation is the
bottleneck (seconds–tens of seconds, measured in Phase 0); retarget+write are
cheap CPU work.
"""
from __future__ import annotations

import itertools
import threading
from collections import defaultdict, deque
from pathlib import Path

import config
import prompts


class ClipPool:
    def __init__(self, runners: list):
        self.runners = runners  # one HYMotionRunner per GPU
        self.ready: dict[str, deque[Path]] = defaultdict(deque)
        self._variant = defaultdict(int)  # per-emotion prompt round-robin
        self._lock = threading.Lock()
        self._seed = itertools.count(1)
        self._stop = threading.Event()

    # --- live path ---------------------------------------------------------
    def next_clip(self, emotion: str) -> Path | None:
        """Pop a ready clip for `emotion` (None if pool momentarily empty)."""
        with self._lock:
            dq = self.ready.get(emotion)
            return dq.popleft() if dq else None

    def depth(self) -> dict[str, int]:
        with self._lock:
            return {e: len(self.ready.get(e, ())) for e in prompts.POOL_EMOTIONS}

    # --- generation --------------------------------------------------------
    def generate_one(self, emotion: str, runner, prompt: str | None = None) -> Path:
        """Generate+retarget+write ONE clip. Blocking (call on a worker).

        TODO(PC-B): wire to the real modules:
            npz = runner.generate(prompt, seed=next(self._seed))
            motion = smpl_to_mmd.retarget(npz, str(config.PMX_PATH))
            out = config.CACHE_DIR / f"{emotion}_{...}.vmd"
            vmd_writer.write_vmd(motion, str(out)); return out
        """
        if prompt is None:
            v = self._variant[emotion]
            self._variant[emotion] += 1
            prompt = prompts.prompt_for(emotion, v)
        raise NotImplementedError("TODO(PC-B): generate->retarget->write one clip")

    # --- background refill -------------------------------------------------
    def start_background_refill(self) -> None:
        """Spawn one filler thread per worker that tops up shallow emotions."""
        for i, runner in enumerate(self.runners):
            threading.Thread(target=self._refill_loop, args=(runner,), daemon=True,
                             name=f"pool-filler-{i}").start()

    def _refill_loop(self, runner) -> None:
        while not self._stop.is_set():
            emotion = self._shallowest()
            if emotion is None:
                self._stop.wait(0.2)
                continue
            try:
                clip = self.generate_one(emotion, runner)
                with self._lock:
                    self.ready[emotion].append(clip)
            except NotImplementedError:
                self._stop.wait(1.0)  # not wired yet (scaffold)
            except Exception as e:  # noqa: BLE001
                print(f"[pool] refill error ({emotion}): {e}")
                self._stop.wait(0.5)

    def _shallowest(self) -> str | None:
        with self._lock:
            short = [e for e in prompts.POOL_EMOTIONS
                     if len(self.ready.get(e, ())) < config.POOL_PER_EMOTION]
        return short[0] if short else None

    def stop(self) -> None:
        self._stop.set()
