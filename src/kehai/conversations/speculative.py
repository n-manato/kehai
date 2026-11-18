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
import copy
import time
from dataclasses import dataclass, field

from loguru import logger

from ..chat_history_manager import (
    history_length,
    replace_message_content,
    truncate_history,
)

# omni mode, audio-first: the speculative run starts on the pause with the
# utterance AUDIO, before Whisper's segment transcript exists (~0.43 s). This
# stands in for the user's words in memory and history until set_text() swaps
# the transcript in; the LLM never sees it (the audio part replaces the text).
SPOKEN_PLACEHOLDER = "[spoken — transcript pending]"


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

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def has_audio(self) -> bool:
        """Whether the held reply already contains voice (its first PCM chunk)."""
        return any('"audio-stream-chunk"' in t for t in self._held)

    async def send_text(self, text: str) -> None:
        async with self._lock:
            if self._open:
                await self._ws.send_text(text)
            else:
                self._held.append(text)

    async def release(self) -> None:
        # Under the lock, so a send racing the flush lands after it, not before.
        async with self._lock:
            had_audio = any('"audio-stream-chunk"' in t for t in self._held)
            for text in self._held:
                await self._ws.send_text(text)
            self._held.clear()
            self._open = True
        if had_audio:
            # Her voice reaches the client NOW: this is the turn's first audio.
            # (tts_manager does not time a reply while it is gated.)
            from ..utils.turn_timing import turn_timing

            turn_timing.mark("first_pcm")
            turn_timing.finish_first_audio(note="speculative HIT, audio held until the endpoint")
        else:
            from ..utils.turn_timing import turn_timing

            turn_timing.mark("released")


@dataclass
class Speculation:
    text: str
    task: asyncio.Task
    gate: GatedSocket
    context: object
    mem_len: int
    hist_len: int
    # The memory's last message as it was before the run. A turn that folds
    # into an unanswered one rewrites that message instead of appending (see
    # BasicMemoryAgent._add_message), and cutting back to mem_len cannot undo a
    # rewrite: the discarded text stayed in the memory and was sent with the
    # next request (2026-10-23 20:21:57, and 10-22 22:39:50).
    mem_tail: dict | None = None
    # The memory at the checkpoint BY IDENTITY: (id of the list, ids of its
    # message dicts, the tail dict). discard() removes the messages added since
    # instead of cutting to mem_len: a length cut also removed whatever another
    # writer had appended meanwhile, and broke as soon as anything earlier in
    # the list was removed (BasicMemoryAgent.tidy_memory, 2026-11-04).
    mem_ref: tuple | None = None
    # speech_epoch when it started: the turn it belongs to. A finalize still
    # waiting on a slow ASR must not take (or drop) the NEXT turn's.
    epoch: int = -1
    # Whether a real reply (not a speculation) was still running when this one
    # started and superseded it — what the rapid-follow-up merge asks about.
    superseded_reply: bool = False
    started: float = field(default_factory=time.monotonic)
    # audio-first (omni): `text` is SPOKEN_PLACEHOLDER until the transcript
    # arrives; text_ready is set once set_text() has swapped it in (or the
    # run was dropped), so the endpoint's HIT test compares words.
    placeholder: bool = False
    text_ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Released at the endpoint before its transcript arrived (fast release,
    # websocket_handler._finalize_vad_turn): the ASR result then only feeds the
    # chat panel / history, or retracts the reply if it turns out to be noise.
    released: bool = False
    # The ears' verdict on this run's audio (omni_addressee.AUDIO): a task
    # started at the pause, read at the endpoint. None = no judge.
    ears: "asyncio.Future | None" = None
    # What the Thinker heard and whether it stops mid-sentence
    # (omni_addressee.COMPLETE.judge_audio): (unfinished | None, heard).
    heard: "asyncio.Future | None" = None
    # Who is talking (active_speaker.speaking, run on the audio so far):
    # (verdict, why, facing) — read at the endpoint.
    asd: "asyncio.Future | None" = None
    # The audio the run started with (base64 WAV), for a check that is only
    # asked when needed (the ears, websocket_handler._addressee_policy).
    audio: str | None = None


def checkpoint(context) -> tuple[int, int, dict | None, tuple | None]:
    """(agent memory length, history file length, the memory's last message,
    the memory by identity) before a speculative run."""
    agent = context.agent_engine
    tidy = getattr(agent, "tidy_memory", None)
    if tidy is not None:
        tidy()
    memory = getattr(agent, "_memory", None)
    mem_len = len(memory) if isinstance(memory, list) else -1
    mem_tail = copy.deepcopy(memory[-1]) if isinstance(memory, list) and memory else None
    mem_ref = (
        (id(memory), frozenset(id(m) for m in memory), memory[-1] if memory else None)
        if isinstance(memory, list)
        else None
    )
    hist_len = (
        history_length(context.character_config.conf_uid, context.history_uid)
        if context.history_uid
        else 0
    )
    return mem_len, hist_len, mem_tail, mem_ref


def set_text(spec: Speculation, text: str) -> None:
    """The transcript of an audio-first run arrived: put the real words where
    the placeholder went (agent memory, history file) and unblock the HIT test."""
    old, spec.text = spec.text, text
    agent = spec.context.agent_engine
    swap = getattr(agent, "replace_user_text", None)
    swapped = bool(swap(old, text)) if swap is not None else False
    if spec.context.history_uid:
        replace_message_content(
            spec.context.character_config.conf_uid, spec.context.history_uid, "human", old, text
        )
    spec.placeholder = False
    spec.text_ready.set()
    if not swapped:
        logger.debug("[spec] placeholder not found in memory (run not stored yet?)")


async def discard(spec: Speculation, reason: str) -> None:
    """Cancel an unconfirmed reply and undo what it wrote."""
    if not spec.task.done():
        spec.task.cancel()
        await asyncio.wait([spec.task], timeout=3)
        # A cancel can be absorbed on the way down (the task then runs on). A
        # dropped run never reached the viewer, so it went on to wait for a
        # "frontend-playback-complete" that never comes — forever, and with it
        # every later voice read as "she is still answering" and was ignored
        # (2026-11-12 12:15:59 -> 23:20, 11 hours). Cancel until it is gone.
        for _ in range(5):
            if spec.task.done():
                break
            logger.warning("[spec] a dropped run survived its cancel — cancelling again")
            spec.task.cancel()
            await asyncio.wait([spec.task], timeout=1)
    agent = spec.context.agent_engine
    memory = getattr(agent, "_memory", None)
    # A compaction or a history reload during the run replaces the list
    # wholesale; then nothing of the checkpoint applies and cutting would
    # delete real turns.
    if isinstance(memory, list) and spec.mem_ref is not None:
        list_id, before, tail = spec.mem_ref
        if id(memory) == list_id:
            memory[:] = [m for m in memory if id(m) in before]
            # A turn that folded into an unanswered one rewrote that message
            # in place: put its old content back.
            if tail is not None and spec.mem_tail is not None and any(m is tail for m in memory):
                tail.clear()
                tail.update(spec.mem_tail)
    elif isinstance(memory, list) and 0 <= spec.mem_len <= len(memory):
        agent._memory = memory[: spec.mem_len]
        if spec.mem_tail is not None and agent._memory:
            agent._memory[-1] = spec.mem_tail
    spec.text_ready.set()  # nobody should wait on a dropped run
    removed = 0
    if spec.context.history_uid:
        removed = truncate_history(
            spec.context.character_config.conf_uid, spec.context.history_uid, spec.hist_len
        )
    logger.info(
        f"[spec] dropped ({reason}) after {(time.monotonic() - spec.started) * 1000:.0f} ms: "
        f"{spec.text[:40]!r}, {spec.gate.held} msgs discarded, {removed} history entries undone"
    )
