"""Speculative replies: start answering on the pause, publish on the endpoint.

The end-of-turn wait is the largest single cost in the felt response time:
~0.9 s of silence before the server-VAD calls the turn over (deliberately long —
this speaker pauses mid-sentence, see conf.yaml end_misses), and only THEN do
LLM + TTS start (~0.55 s more). But the transcript is usually final long before
the wait is: the VAD closes the speech segment after ~0.38 s and its ASR is
back a few hundred ms later.

So the reply starts on that segment — the real conversation pipeline, unchanged
— with everything it sends held back (GatedSocket). When the endpoint fires on
the same transcript the held output is released at once, already part-way
generated. If the user speaks again, or the final transcript differs, the run is
cancelled and every trace of it is undone: the agent's in-memory history and
the history file are cut back to the checkpoint taken before it started, and the
client, having received nothing, never knew. Interruption behaviour is exactly
what it was; only the latency on a turn that really did end changes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from loguru import logger

from ..chat_history_manager import history_length, truncate_history


class GatedSocket:
    """Stands in for the client WebSocket while a reply is speculative: every
    send is queued until release(), then flushed in order and passed through."""

    def __init__(self, websocket):
        self._ws = websocket
        self._held: list[str] = []
        self._open = False
        self._lock = asyncio.Lock()

    @property
    def held(self) -> int:
        return len(self._held)

    async def send_text(self, text: str) -> None:
        async with self._lock:
            if self._open:
                await self._ws.send_text(text)
            else:
                self._held.append(text)

    async def release(self) -> None:
        # Under the lock, so a send racing the flush lands after it, not before.
        async with self._lock:
            for text in self._held:
                await self._ws.send_text(text)
            self._held.clear()
            self._open = True


@dataclass
class Speculation:
    text: str
    task: asyncio.Task
    gate: GatedSocket
    context: object
    mem_len: int
    hist_len: int
    started: float = field(default_factory=time.monotonic)


def checkpoint(context) -> tuple[int, int]:
    """(agent memory length, history file length) before a speculative run."""
    memory = getattr(context.agent_engine, "_memory", None)
    mem_len = len(memory) if isinstance(memory, list) else -1
    hist_len = (
        history_length(context.character_config.conf_uid, context.history_uid)
        if context.history_uid
        else 0
    )
    return mem_len, hist_len


async def discard(spec: Speculation, reason: str) -> None:
    """Cancel an unconfirmed reply and undo what it wrote."""
    if not spec.task.done():
        spec.task.cancel()
        await asyncio.wait([spec.task], timeout=3)
    agent = spec.context.agent_engine
    memory = getattr(agent, "_memory", None)
    # A compaction during the run replaces the list wholesale; then the
    # checkpoint length means nothing and cutting would delete real turns.
    if isinstance(memory, list) and 0 <= spec.mem_len <= len(memory):
        agent._memory = memory[: spec.mem_len]
    removed = 0
    if spec.context.history_uid:
        removed = truncate_history(
            spec.context.character_config.conf_uid, spec.context.history_uid, spec.hist_len
        )
    logger.info(
        f"[spec] dropped ({reason}) after {(time.monotonic() - spec.started) * 1000:.0f} ms: "
        f"{spec.text[:40]!r}, {spec.gate.held} msgs discarded, {removed} history entries undone"
    )
