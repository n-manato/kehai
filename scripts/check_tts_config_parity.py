"""The two streaming-TTS configs must be interchangeable where tts_manager touches them.

This exists because of a bug that cost a debugging session: ElevenStreamConfig
lacked `text_lang`, which tts_manager reads on a path only longer replies reach.
Short test sentences never got there, so the wiring looked fine — and the real
replies died mid-stream with AttributeError, went silent, and were written to
chat history with empty content. Duck typing made the mismatch invisible until
runtime.

So the contract is pinned here instead, and deliberately not as a hand-written
list: every attribute the manager reads off the config is discovered from the
manager's SOURCE. Add a backend, or read a new attribute in the manager, and
this fails immediately rather than on the first long sentence.

Run: .venv/Scripts/python.exe scripts/check_tts_config_parity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kehai.tts.elevenlabs_stream import ElevenStreamConfig  # noqa: E402
from kehai.tts.gsv_stream import GSVStreamConfig  # noqa: E402

MANAGER = ROOT / "src/kehai/conversations/tts_manager.py"

# Read only INSIDE a branch that has already established which backend it holds,
# so these are legitimately one-sided. Everything else is on the shared path.
GSV_ONLY = {"base_url"}
ELEVEN_ONLY = {"model_id"}


def main() -> int:
    src = MANAGER.read_text(encoding="utf-8")
    reads = sorted(set(re.findall(r"_stream_cfg\.([A-Za-z_][A-Za-z0-9_]*)", src)))
    print("tts_manager reads off _stream_cfg:", ", ".join(reads))

    failures = []
    for cfg, only in (
        (ElevenStreamConfig(), GSV_ONLY),
        (GSVStreamConfig(), ELEVEN_ONLY),
    ):
        missing = [a for a in reads if a not in only and not hasattr(cfg, a)]
        name = type(cfg).__name__
        print(f"  {name}: {'ok' if not missing else 'MISSING ' + str(missing)}")
        if missing:
            failures.append(
                f"{name} lacks {missing} — the manager reads it, so a reply that "
                "reaches that line raises mid-sentence and goes silent"
            )

    # Existing is not enough: it has to carry the language.
    for cfg in (ElevenStreamConfig(), GSVStreamConfig()):
        if not cfg.text_lang:
            failures.append(f"{type(cfg).__name__}.text_lang is empty")

    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    if failures:
        return 1
    print("PASS: both stream configs satisfy every attribute the manager reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
