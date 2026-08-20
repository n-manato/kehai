"""Resident HY-Motion inference wrapper.

Keep the billion-param model LOADED in VRAM (never reload per request). With
CUDA_VISIBLE_DEVICES=0,1 we run one worker per GPU (see server.py / pool.py).

HY-Motion ships only `local_infer.py` (CLI) and `gradio_app.py` — no library
API. So we import/lift the inference internals from the HY-Motion-1.0 repo and
call them in-process. `# TODO(PC-B)` marks where to wire the real model.
"""
from __future__ import annotations

import time
from pathlib import Path

import config


class HYMotionRunner:
    def __init__(self, device: str = "cuda:0", ckpts_dir: Path = config.CKPTS_DIR):
        self.device = device
        self.ckpts_dir = Path(ckpts_dir)
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load HY-Motion weights onto self.device and keep resident.

        TODO(PC-B): import the model builder from the HY-Motion-1.0 repo
        (the same path local_infer.py uses), load `self.ckpts_dir`, move to
        `self.device`, set eval(). Optionally load Text2MotionPrompter for
        duration estimation / prompt rewrite (can be disabled for speed).
        """
        raise NotImplementedError("TODO(PC-B): load HY-Motion model resident")

    def generate(self, prompt: str, duration: float | None = None, seed: int = 0):
        """Run text->motion. Returns path to a saved SMPL NPZ.

        Caps duration to config.MAX_DURATION_SEC. Logs wall-clock latency — the
        Phase-0 number that sizes the whole pool design.

        TODO(PC-B): call the lifted inference (DiT + flow-matching sampling) with
        num_seeds=1; save SMPL/SMPL-H NPZ to config.CACHE_DIR; return its path.
        """
        dur = min(duration or config.DEFAULT_DURATION_SEC, config.MAX_DURATION_SEC)
        t0 = time.perf_counter()
        # _npz = self._sample(prompt, dur, seed)  # TODO(PC-B)
        dt = time.perf_counter() - t0
        print(f"[hy-motion] gen '{prompt[:40]}' dur={dur:.1f}s -> {dt:.1f}s on {self.device}")
        raise NotImplementedError("TODO(PC-B): run sampling + save NPZ, return path")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HY-Motion runner self-test (Phase-0)")
    ap.add_argument("--selftest", default="a person waving hello")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    r = HYMotionRunner(device=args.device)
    r.load()
    npz = r.generate(args.selftest, duration=args.duration)
    print("NPZ:", npz)  # inspect keys/shape + the printed latency
