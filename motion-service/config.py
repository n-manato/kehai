"""motion-service configuration (env-overridable).

Keep this dependency-free so it imports cheaply from any module.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- paths -----------------------------------------------------------------
# HY-Motion checkpoints (git-ignored; download on PC-B).
CKPTS_DIR = Path(os.getenv("HYM_CKPTS", ROOT / "ckpts" / "tencent" / "HY-Motion-1.0"))
# SMPL/SMPL-H body model files (git-ignored; license-restricted).
BODY_MODELS_DIR = Path(os.getenv("HYM_BODY_MODELS", ROOT / "body_models"))
# Target MMD model whose rest pose / bone names drive retargeting.
PMX_PATH = Path(os.getenv("HYM_PMX", ROOT / "assets" / "yyb_miku.pmx"))
# Generated/retargeted .vmd cache (git-ignored).
CACHE_DIR = Path(os.getenv("HYM_CACHE", ROOT / "cache"))

# --- inference / parallelism ----------------------------------------------
# One worker pinned per GPU. With CUDA_VISIBLE_DEVICES=0,1 -> 2 workers.
GPUS = [g for g in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""]
NUM_WORKERS = max(1, len(GPUS))
# HY-Motion best-practice caps.
MAX_DURATION_SEC = float(os.getenv("HYM_MAX_DURATION", "5.0"))
DEFAULT_DURATION_SEC = float(os.getenv("HYM_DEFAULT_DURATION", "4.0"))
TARGET_FPS = int(os.getenv("HYM_FPS", "30"))

# --- pool ------------------------------------------------------------------
# How many ready clips to keep warm per emotion. Tune after Phase-0 latency
# measurement: deeper pool hides slower generation.
POOL_PER_EMOTION = int(os.getenv("HYM_POOL_PER_EMOTION", "3"))

# --- server ----------------------------------------------------------------
HOST = os.getenv("HYM_HOST", "0.0.0.0")
PORT = int(os.getenv("HYM_PORT", "8800"))


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
