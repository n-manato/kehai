"""Bridge: tool execution (deep in the agent/MCP stack) -> the CURRENT turn's
voice pipeline.

A tool-using reply is silent until the whole decide-tool -> execute -> re-prompt
loop finishes (measured 6-30s of dead air — "ツールを使うとしゃべらなくなる").
single_conversation registers a per-turn callback here; the tool executor fires
it when the FIRST tool starts, and the callback speaks a short canned filler
("ちょっと調べてみるね!") through the normal TTS pipeline — the ChatGPT-Voice
acknowledgement pattern. Fired at most once per turn; cleared in the turn's
finally so a stale callback can never speak into a later turn.
"""

import asyncio
from typing import Awaitable, Callable, Optional

_callback: Optional[Callable[[str], Awaitable[None]]] = None


def set_callback(cb: Callable[[str], Awaitable[None]]) -> None:
    """Register the current turn's filler-speaker (overwrites any stale one)."""
    global _callback
    _callback = cb


def clear() -> None:
    global _callback
    _callback = None


def fire(tool_name: str) -> None:
    """Called by the tool executor when a tool starts. Fire-and-forget; never
    raises into the tool path."""
    global _callback
    cb = _callback
    if cb is None:
        return
    _callback = None  # once per turn
    try:
        asyncio.get_running_loop().create_task(cb(tool_name))
    except RuntimeError:  # no running loop (sync test context) — skip quietly
        pass
