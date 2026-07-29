import asyncio
import base64
import io
import json
import os
import re
import time
import uuid
import wave
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger

import httpx

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..realtime import webrtc_manager
from ..utils.turn_timing import turn_timing
from ..tts.tts_interface import TTSInterface
from ..tts.gsv_stream import (
    GSVStreamConfig,
    stream_sentence_pcm,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH,
    PCM_CHANNELS,
)
from ..utils.stream_audio import prepare_audio_payload
from ..utils.speech_duration import estimate_speech_duration
from .types import WebSocketSend
from .. import motion_cache
from ..motion import actionplan_client, gesture_client, motion_lib
from ..motion.prompts import emotion_from_keyword, prompt_for_key

# HYBRID (KAWAII + GestureLSM): KAWAII talk-loop plays INSTANTLY on every sentence; on
# LONG, non-emote sentences we ALSO generate audio-driven GestureLSM co-speech and
# crossfade it OVER the loop (actions.motion.upgrade_id) once it lands — natural,
# audio-synced variety on the sentences where it reads, without touching latency or the
# cute emote poses. GestureLSM needs >=6s audio; short/emote sentences stay pure KAWAII.
# GestureLSM covers almost every real talking sentence now (2.5s bar; shorter clauses
# get padded to the gateway's 6s min). Only quick fillers and emote sentences stay
# KAWAII. Combined with the frontend keeping GestureLSM CONTINUOUS across sentences
# (it doesn't drop back to the KAWAII loop between them), this makes GestureLSM the
# dominant driver (~70%): KAWAII is the ~2s startup lead-in + the cute emote poses.
# Tune live: HYBRID_GESTURELSM_MIN_SECS (raise = more KAWAII/anime, lower = more gen).
_HYBRID_GESTURELSM_MIN_SECS = float(os.getenv("HYBRID_GESTURELSM_MIN_SECS", "2.5"))
# EARLY-FIRE window: fire GestureLSM from the FIRST ~1.2s of streamed audio (not at
# sentence-end) so it generates DURING the rest of synthesis and lands ~2s into
# playback — early enough to crossfade over the KAWAII loop before the frontend's
# stale-drop rejects it. Without this, GestureLSM lands 2.4-4.4s in and only ever
# applies on LONG sentences (the "GestureLSM doesn't show on normal speech" issue).
_HYBRID_EARLY_WIN_S = float(os.getenv("HYBRID_GESTURELSM_EARLY", "1.2"))
# OPENER fast-path: the turn's FIRST GestureLSM has zero lead time (its sentence is
# already playing while it generates), so waiting for 1.2s of audio pushes the takeover
# to ~2.5-3.5s. Fire it from the very FIRST PCM chunk instead (~0.35s window — the pad
# to GESTURE_MIN_SECONDS supplies the rest; the gateway trims to `dur`). Sentences 2+
# generate during the PREVIOUS sentence, so they keep the higher-fidelity 1.2s window.
# Voice is untouched: the fire is fire-and-forget on a separate service slot.
_HYBRID_FIRST_WIN_S = float(os.getenv("HYBRID_GESTURELSM_FIRST", "0.35"))


def _is_emote_sentence(actions, text: str) -> bool:
    """This sentence resolves to a cute KAWAII one-shot pose (tag or JP keyword cue),
    so it stays pure KAWAII and is NOT handed to GestureLSM."""
    return bool(
        motion_lib.is_emote(getattr(actions, "gesture", None))
        or motion_lib.emote_for_text(text)
    )


def _wants_gesturelsm_upgrade(actions, text: str, secs: float) -> bool:
    return (
        gesture_client.enabled()
        and secs >= _HYBRID_GESTURELSM_MIN_SECS
        and not _is_emote_sentence(actions, text)
    )


# Opt-in true TTS streaming: stream GSV raw PCM to the frontend chunk-by-chunk so
# the first phrase plays ~1s sooner than waiting for the whole sentence's WAV.
# Enabled with GSV_STREAM=1 (base URL overridable via GSV_STREAM_URL). When off,
# the classic file-based path below is used unchanged.
_STREAM_ENABLED = os.getenv("GSV_STREAM", "0") == "1"
# Co-speech gesture pacing (streaming). GestureLSM makes ONE ~6s clip per call, but
# streaming feeds it a sentence at a time; firing per sentence (~2-3s) asks for 2-3x
# more clips than it can generate during a bursty reply, so they queue and never
# land in time. Fire at most one gesture per this many seconds of speech — a clip
# loops/crossfades to cover the span — so generation keeps pace with playback.
_GESTURE_MIN_SPACING_S = float(os.getenv("GSV_STREAM_GESTURE_SPACING", "5.0"))
# D2 (early-fire): once this many seconds of a sentence's PCM have streamed in,
# fire GestureLSM on that window IMMEDIATELY instead of waiting for the
# sentence's synthesis to end — the ~1.5-2.8s generation overlaps the rest of
# the stream, so the clip is ready (and crossfades over the procedural talking
# body) that much sooner.
_GESTURE_EARLY_WINDOW_S = float(os.getenv("GSV_STREAM_GESTURE_EARLY", "2.5"))
# The TURN'S FIRST gesture fires from a much smaller window so the real
# co-speech clip lands near the START of the reply (the procedural talking body
# only has to cover ~2s instead of ~4-5s). GestureLSM pads short audio to its
# canvas, so 1.0s is enough context for an opening gesture.
_GESTURE_FIRST_WINDOW_S = float(os.getenv("GSV_STREAM_GESTURE_FIRST", "1.0"))
# TEXT-streaming (ActionPlan) fire pacing: one generation takes ~1-2s, so firing
# EVERY sentence of a bursty short-sentence reply (synthesis runs ~2x faster than
# playback) builds a FIFO backlog at the single-GPU service (measured 15-19s —
# clips arrived long after their sentences played). Fire at most one clip per
# this many ESTIMATED speech seconds; skipped short sentences are covered by the
# previous clip's est+1.5s tail + last-frame hold (chained continuity).
# 4.0: at 2.0 every ~2.6s-est sentence fired its own 4-5s clip, so clips
# replaced each other after 1-2s (arm-raise -> arm-raise -> ... = "モーションが
# バグり散らかす") and the single-GPU service queue saturated again (+5s late).
# One clip per ~4s of speech lets each gesture actually play out; chaining keeps
# the seams continuous.
_TEXT_GESTURE_SPACING_S = float(os.getenv("MOTION_TEXT_SPACING", "4.0"))

# ONE process-wide HTTP client for the streaming TTS: TTSTaskManager is created
# PER TURN, so a per-instance client re-did the TLS handshake to the gateway on
# every turn's first sentence (~0.2-0.4s of the felt TTFA). A shared client
# keeps the connection warm across turns.
_shared_stream_client: Optional[httpx.AsyncClient] = None


def _get_stream_client() -> httpx.AsyncClient:
    global _shared_stream_client
    if _shared_stream_client is None or _shared_stream_client.is_closed:
        _shared_stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        )
    return _shared_stream_client


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw int16 mono PCM as a WAV container (for co-speech gesture, which
    needs a parseable WAV — the streamed chunks themselves are headerless PCM)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(PCM_SAMPLE_WIDTH)
        w.setframerate(PCM_SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# Serialize actual TTS synthesis across the whole app. Sentences are queued as
# parallel tasks for ordered delivery, but most local TTS backends (e.g. a
# single GPT-SoVITS instance) process requests one at a time. Firing them all at
# once made later sentences sit in the backend's queue and blow their per-request
# read timeout before synthesis even started — so long replies stopped speaking
# after the first sentence or two. Generating one at a time means each request's
# timeout only covers its OWN synthesis.
_tts_synthesis_lock = asyncio.Semaphore(1)


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self, stream_cfg: Optional[GSVStreamConfig] = None) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0

        # --- streaming (GSV_STREAM=1) -------------------------------------
        self.streaming = _STREAM_ENABLED
        self._stream_cfg: Optional[GSVStreamConfig] = None
        self._stream_client: Optional[httpx.AsyncClient] = None
        # Send-order chain: a streamed sentence sends start/chunks/end LIVE (not via
        # the payload queue), so sentence N must finish sending before N+1 begins.
        # Each speak() links a fresh future onto the tail; a task awaits its
        # predecessor's future before sending and resolves its own in `finally`
        # (even on cancel), so the chain never deadlocks and clear() just drops the
        # tail to start a fresh chain. Awaiting the predecessor also serializes GSV
        # access (one synth at a time), so no extra lock is needed.
        self._stream_prev_done: Optional[asyncio.Future] = None
        # Gesture pacing: accumulate spoken seconds; fire a co-speech gesture only
        # once per _GESTURE_MIN_SPACING_S (plus one at the very start so the body
        # starts moving early). Reset in clear().
        self._stream_gesture_accum = 0.0
        self._stream_gesture_fired = False
        self._stream_upgrade_n = (
            0  # GestureLSM upgrades fired this turn (opener fast-path)
        )
        # Loudness stabiliser: GSV normalises each cut5 FRAGMENT separately, so the level
        # steps mid-sentence (measured up to 10dB). One instance per manager so the
        # correction carries across sentences (sends are serialised = playback order).
        self._level_stab = None
        self._stream_total_secs = 0.0  # spoken seconds this turn (gesture ledger)
        if self.streaming:
            # Per-character voice profile (ref/langs/weights) comes from the
            # caller (service_context.gsv_stream_config()); a bare default is
            # the built-in Rem profile. base_url always follows GSV_STREAM_URL.
            self._stream_cfg = stream_cfg or GSVStreamConfig()
            self._stream_cfg.base_url = os.getenv(
                "GSV_STREAM_URL", self._stream_cfg.base_url
            )
            # Shared, process-wide client: keeps the gateway TLS connection warm
            # across turns (the manager itself is per-turn).
            self._stream_client = _get_stream_client()
            logger.info(
                f"[tts_manager] TRUE TTS STREAMING enabled -> {self._stream_cfg.base_url}"
            )

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if self.streaming:
            # All streaming logic (ordering, empty-text, gesture) lives in the
            # streaming methods; the classic file-based path below is untouched.
            prev = self._stream_prev_done
            done = asyncio.get_event_loop().create_future()
            self._stream_prev_done = done
            task = asyncio.create_task(
                self._process_tts_stream(
                    tts_text=tts_text,
                    display_text=display_text,
                    actions=actions,
                    prev_done=prev,
                    my_done=done,
                    websocket_send=websocket_send,
                )
            )
            self.task_list.append(task)
            return

        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Fire bespoke body-motion generation NOW — before the TTS task even
        # starts — so it overlaps both the TTS semaphore wait AND synthesis (the
        # largest head-start available). This is fire-and-forget: it returns an id
        # synchronously and never awaits, so ordered delivery and never-delay-speech
        # both hold. The frontend fetches /motion/clip/{id} at speech start and
        # falls back to the fresh pool buffer if it isn't ready.
        # gesture_client (audio-driven GestureLSM) is fired later, from _process_tts,
        # because it needs the finished audio. Only fire the text-driven HY-Motion
        # path here when gesture is OFF, so the two never double-drive actions.motion.
        if (
            actions is not None
            and motion_cache.enabled()
            and not gesture_client.enabled()
        ):
            # Prefer the CONTENT gesture the LLM tagged (e.g. "point", "count");
            # fall back to the sentence's emotion. This is the body-motion KEY.
            key = actions.gesture or emotion_from_keyword(
                actions.emotions[0] if actions.emotions else None
            )
            dur = estimate_speech_duration(tts_text)
            mid = motion_cache.start_generation(key, dur, prompt_for_key(key))
            actions.motion = {"id": mid, "key": key, "duration": dur}

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    def _level(self, pcm: bytes) -> bytes:
        """Even out GSV's per-fragment loudness steps. OFF BY DEFAULT: an A/B on the live
        stream showed it does NOT measurably flatten real speech (worst step 5.1-6.9dB
        off vs 6.3-6.9dB on), so the voice path stays untouched until the artefact is
        pinned down. Enable to experiment: TTS_LEVEL_STABILIZE=1."""
        if os.getenv("TTS_LEVEL_STABILIZE", "0") != "1":
            return pcm
        try:
            if self._level_stab is None:
                from ..utils.level_stabilizer import LevelStabilizer

                self._level_stab = LevelStabilizer()
            return self._level_stab.feed(pcm)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[tts] level stabiliser bypassed: {e!r}")
            return pcm

    def _level_flush(self) -> bytes:
        """Sub-frame tail held by the stabiliser (<20ms), so each sentence is complete."""
        try:
            return self._level_stab.flush() if self._level_stab is not None else b""
        except Exception:  # noqa: BLE001
            return b""

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.
        """
        buffered_payloads: Dict[int, Dict] = {}

        while True:
            try:
                # Get payload from queue
                payload, sequence_number = await self._payload_queue.get()
                buffered_payloads[sequence_number] = payload

                # Send payloads in order
                while self._next_sequence_to_send in buffered_payloads:
                    next_payload = buffered_payloads.pop(self._next_sequence_to_send)
                    await websocket_send(json.dumps(next_payload))
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            # Audio-driven co-speech gesture: now that the speech audio exists, fire
            # GestureLSM on IT (fire-and-forget) and hand the frontend the clip id.
            # The audio payload is queued immediately below — gesture never delays it.
            if actions is not None and gesture_client.enabled() and audio_file_path:
                # Skip GestureLSM on junk fragments the sentence splitter emits as
                # standalone "sentences" — bare list markers ("2.", "3."), lone
                # punctuation/numbers. They carry no speakable/gestureable content, so
                # generating a gesture just wastes a gateway+GPU round-trip. Require at
                # least one letter / kana / kanji. The frontend's warm loop (or the
                # prior gesture) keeps the body moving over the fragment meanwhile.
                if not re.search(r"[a-zA-Z぀-ヿ一-鿿]", tts_text):
                    logger.info(f"[gesture] skip no-content fragment: {tts_text!r}")
                else:
                    try:
                        if actionplan_client.enabled():
                            # TEXT-driven (ActionPlan): no audio needed.
                            _emo = (getattr(actions, "emotions", None) or [None])[0]
                            _mid = motion_cache.start_gesture_text(
                                tts_text,
                                _emo,
                                # clip ~= sentence length (was +1.5): shorter = faster
                                # gen + less overlap/bleed into the next sentence.
                                estimate_speech_duration(tts_text) + 0.8,
                                session=f"tts-{id(self)}",  # per-turn chain
                                gesture=getattr(actions, "gesture", None),
                            )
                        else:
                            with open(audio_file_path, "rb") as _f:
                                _audio_bytes = _f.read()
                            _mid = motion_cache.start_gesture(_audio_bytes, tts_text)
                        actions.motion = {
                            "id": _mid,
                            "key": "gesture",
                            "duration": estimate_speech_duration(tts_text),
                        }
                    except Exception as _ge:  # noqa: BLE001
                        logger.warning(f"[gesture] fire failed: {_ge}")
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Queue the payload with its sequence number
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _process_tts_stream(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        prev_done: Optional[asyncio.Future],
        my_done: asyncio.Future,
        websocket_send: WebSocketSend,
    ) -> None:
        """Stream one sentence's raw PCM to the frontend as GSV produces it. Sends,
        in order across sentences (via the future chain):
            audio-stream-start -> audio-stream-chunk* -> audio-stream-end
        An empty / punctuation-only sentence sends start+end with no chunks so the
        display text + expressions still show and the chain still advances.
        Co-speech gesture is fired from the accumulated PCM at stream end (fire-and-
        forget) and its clip id rides the end frame (frontend polls /motion/clip)."""
        dt = (
            display_text.to_dict()
            if isinstance(display_text, DisplayText)
            else display_text
        )
        speakable = len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) > 0
        pcm_parts: list[bytes] = []
        try:
            # Wait our turn in the send order (predecessor finished sending).
            if prev_done is not None:
                try:
                    await prev_done
                except Exception:  # noqa: BLE001 — predecessor error/cancel: still our turn
                    pass

            # Phase G: when the client negotiated a WebRTC peer, its outbound
            # track carries the PCM (browser-native jitter buffer + true AEC
            # reference) and the WS only carries the control frames. `play_in`
            # tells the frontend how far ahead this sentence is queued on the
            # track (its subtitle/emotion/gesture fire that many seconds later,
            # restoring the sync the WS player's origin clock used to provide).
            rtc_track = webrtc_manager.track_for_send(websocket_send)
            has_content = bool(re.search(r"[a-zA-Z぀-ヿ一-鿿]", tts_text))

            # TEXT-driven streaming gesture (ActionPlan): fire BEFORE the start
            # frame goes out, so the clip id rides audio-stream-START and the
            # frontend can schedule the apply AT the sentence's play origin —
            # riding only the end frame applied it ~1s into the voice ("若干
            # モーションが遅い"). Generation needs no audio: first frames land
            # ~0.6s after this line, right at the first spoken frame.
            text_gesture = actionplan_client.enabled()
            _gest_go: Optional[asyncio.Event] = None  # set at first TTS chunk
            if (
                text_gesture
                and speakable
                and has_content
                and actions is not None
                and gesture_client.enabled()
            ):
                try:
                    _est = estimate_speech_duration(tts_text)
                    # Pace the fires (see _TEXT_GESTURE_SPACING_S): a skipped
                    # short sentence still counts toward the next fire, and its
                    # body is covered by the previous clip's tail + hold.
                    _due = (not self._stream_gesture_fired) or (
                        self._stream_gesture_accum + _est >= _TEXT_GESTURE_SPACING_S
                    )
                    if _due:
                        _emo = (getattr(actions, "emotions", None) or [None])[0]
                        _sid = str(
                            id(getattr(websocket_send, "__self__", websocket_send))
                        )
                        # Defer the GPU launch to the first TTS chunk: ActionPlan
                        # and GSV share PC-B GPU1, and launching at fire time
                        # slowed the VOICE (tts_first_chunk 500->1109ms). The id
                        # still rides the start frame; only the network/GPU work
                        # waits. Voice speed always wins.
                        _gest_go = asyncio.Event()
                        mid = motion_cache.start_gesture_text(
                            tts_text,
                            _emo,
                            _est
                            + 0.8,  # clip ~= sentence length (was +1.5): less bleed
                            session=_sid,
                            start_event=_gest_go,
                            gesture=getattr(actions, "gesture", None),
                            # Turn opener -> serve the pre-generated GestureLSM bridge
                            # instead of a KAWAII loop, so the ~1.4s before the real
                            # (audio-driven) clip lands is already GestureLSM-styled.
                            prefer_bridge=not self._stream_gesture_fired,
                            # Content-matched intent LLM on EVERY fire incl. the first
                            # (the emotion heuristic used to leave the opener generic).
                            # It runs on GPU0 during the deferred-launch wait, so the
                            # VOICE is unaffected — only the gesture may land a bit later.
                            use_intent=True,
                        )
                        actions.motion = {
                            "id": mid,
                            "key": "gesture",
                            "duration": _est,
                            # 'talk' loops may be SKIPPED by the frontend while GestureLSM
                            # is already driving (keeps GestureLSM continuous); 'emote'
                            # always plays (the cute pose is the point).
                            "kind": "emote"
                            if _is_emote_sentence(actions, tts_text)
                            else "talk",
                        }
                        self._stream_gesture_fired = True
                        self._stream_gesture_accum = 0.0
                        logger.info(
                            f"[gesture-ledger] FIRE(text) est {_est:.1f}s emo={_emo}"
                        )
                    else:
                        self._stream_gesture_accum += _est
                except Exception as _ge:  # noqa: BLE001
                    logger.warning(f"[gesture] text fire failed: {_ge}")

            await websocket_send(
                json.dumps(
                    {
                        "type": "audio-stream-start",
                        "sample_rate": PCM_SAMPLE_RATE,
                        "display_text": dt,
                        "actions": actions.to_dict() if actions else None,
                        "forwarded": False,
                        "play_in": (
                            round(rtc_track.buffered_seconds(), 3)
                            if rtc_track is not None
                            else None
                        ),
                    }
                )
            )

            early_fired_secs = 0.0  # >0 => gesture already fired mid-stream (D2)
            _upgrade_fired = False  # HYBRID: GestureLSM upgrade already early-fired
            _t_first_pcm = (
                0.0  # monotonic of this sentence's first PCM (gap ledger ref)
            )
            # Opener fast-path window: the turn's FIRST upgrade fires at the first PCM
            # chunk (no lead time to hide behind); later sentences wait for 1.2s of
            # real audio (better speech-matched motion — they land pre-sentence anyway).
            # No min-duration gate: the first "sentence" is usually a SHORT comma-split
            # clause (faster_first_response), and skipping it pushed GestureLSM to
            # sentence 2 (~2.5-3.5s takeover). The frontend gives the opener's clip a
            # slack clock so it BRIDGES across the clause boundary into sentence 2.
            _up_win = (
                _HYBRID_FIRST_WIN_S
                if self._stream_upgrade_n == 0
                else _HYBRID_EARLY_WIN_S
            )
            turn_timing.mark("first_sentence")  # LLM's first sentence reached TTS
            if speakable and self._stream_cfg is not None:
                # (cut0-per-sentence was tried here and REVERTED: whole-sentence
                # lumps starved the track at sentence boundaries and made the
                # glottal-stop artifact WORSE. cut5 + the track's short re-prime
                # is the working combination — see TTSAudioTrack.REPRIME_SECONDS.)
                #
                # GapCloser: the fine-tuned RemGSV model stochastically bakes
                # 40-170ms silences into the MIDDLE of words (~1/6 renders; the
                # residual 促音). Close them in the outbound PCM; the sentence
                # text guards real geminates (っ/ッ). TTS_GAP_CLOSE=0 disables.
                _gap_closer = None
                if rtc_track is not None and os.getenv("TTS_GAP_CLOSE", "1") == "1":
                    from ..utils.gap_closer import GapCloser

                    _gap_closer = GapCloser(tts_text, lang=self._stream_cfg.text_lang)
                acc_bytes = 0
                async for chunk in stream_sentence_pcm(
                    tts_text, self._stream_cfg, client=self._stream_client
                ):
                    pcm_parts.append(chunk)
                    acc_bytes += len(chunk)
                    if acc_bytes == len(chunk):  # first PCM of this sentence
                        _t_first_pcm = time.monotonic()  # voice can start NOW (gap ref)
                        turn_timing.mark("first_pcm")
                        turn_timing.finish_first_audio()
                        if _gest_go is not None:
                            _gest_go.set()  # voice is out — NOW start motion gen
                    # Even out GPT-SoVITS' per-fragment level steps (measured up to
                    # 10dB WITHIN one sentence) before the audio leaves. Latency-free,
                    # so TTFA and every downstream clock are untouched.
                    if rtc_track is not None:
                        if _gap_closer is not None:
                            out = self._level(_gap_closer.feed(chunk))
                            if out:
                                rtc_track.push_pcm32k(out)
                        else:
                            out = self._level(chunk)
                            if out:
                                rtc_track.push_pcm32k(out)
                    # `chunk` may come back EMPTY (a sub-frame remainder is held for the
                    # next chunk) — skip only the send, never the gesture logic below.
                    elif chunk := self._level(chunk):
                        await websocket_send(
                            json.dumps(
                                {
                                    "type": "audio-stream-chunk",
                                    "data": base64.b64encode(chunk).decode("ascii"),
                                }
                            )
                        )
                    # D2 early-fire: enough of this sentence has streamed — start
                    # GestureLSM on the window NOW so generation overlaps the rest
                    # of the synthesis instead of starting after it.
                    _early_win = (
                        _GESTURE_FIRST_WINDOW_S
                        if not self._stream_gesture_fired
                        else _GESTURE_EARLY_WINDOW_S
                    )
                    if (
                        early_fired_secs == 0.0
                        and not text_gesture  # ActionPlan already fired from text
                        and actions is not None
                        and gesture_client.enabled()
                        and has_content
                        and acc_bytes >= int(_early_win * PCM_SAMPLE_RATE) * 2
                    ):
                        acc_secs = acc_bytes / 2 / PCM_SAMPLE_RATE
                        if (not self._stream_gesture_fired) or (
                            self._stream_gesture_accum + acc_secs
                            >= _GESTURE_MIN_SPACING_S
                        ):
                            try:
                                mid = motion_cache.start_gesture(
                                    _pcm_to_wav_bytes(b"".join(pcm_parts)),
                                    tts_text,
                                    # energetic span + a short settle only:
                                    # the silence-tail of a long clip DECAYS
                                    # (looks frozen while the voice is loud) —
                                    # end the clip early and let the RMS-driven
                                    # procedural layer take over instead.
                                    dur_override=acc_secs + 1.5,
                                )
                                actions.motion = {
                                    "id": mid,
                                    "key": "gesture",
                                    "duration": estimate_speech_duration(tts_text),
                                }
                                self._stream_gesture_fired = True
                                self._stream_gesture_accum = 0.0
                                early_fired_secs = acc_secs
                                logger.info(
                                    "[gesture-ledger] FIRE(early) at "
                                    f"~{self._stream_total_secs + acc_secs:.1f}s of turn "
                                    f"speech (window {acc_secs:.1f}s)"
                                )
                            except Exception as _ge:  # noqa: BLE001
                                logger.warning(f"[gesture] early fire failed: {_ge}")
                    # HYBRID early-fire: KAWAII already plays instantly from text; the
                    # moment ~1.2s of this sentence's audio exists, fire GestureLSM on
                    # that window so it lands ~2s into playback and crossfades over the
                    # loop in time (vs firing at sentence-end, which lands too late and
                    # gets stale-dropped on all but long sentences). Non-emote only.
                    elif (
                        text_gesture
                        and not _upgrade_fired
                        and actions is not None
                        and actions.motion
                        and gesture_client.enabled()
                        and has_content
                        and not _is_emote_sentence(actions, tts_text)
                        and acc_bytes >= int(_up_win * PCM_SAMPLE_RATE) * 2
                    ):
                        _upgrade_fired = True
                        try:
                            _gid = motion_cache.start_gesture(
                                _pcm_to_wav_bytes(b"".join(pcm_parts)),
                                tts_text,
                                # Ask for MORE than the sentence needs: the model caps a
                                # clip at ~4.3s anyway, and a clip trimmed to a short
                                # clause (2.2s) leaves a gap the KAWAII loop fills —
                                # which flips the body between two motion styles
                                # mid-turn. Always take the longest clip on offer.
                                dur_override=max(
                                    estimate_speech_duration(tts_text) + 1.5, 5.0
                                ),
                                t_audio=_t_first_pcm or None,
                            )
                            actions.motion["upgrade_id"] = _gid
                            self._stream_upgrade_n += 1
                            logger.info(
                                f"[gesture-ledger] HYBRID early-upgrade(GestureLSM) "
                                f"{_gid[:8]} at ~{acc_bytes / 2 / PCM_SAMPLE_RATE:.1f}s window"
                            )
                        except Exception as _ge:  # noqa: BLE001
                            logger.warning(
                                f"[gesture] hybrid early upgrade failed: {_ge}"
                            )

            # drain the gap-closer's lookahead tail (sentence is fully streamed)
            if speakable and self._stream_cfg is not None and _gap_closer is not None:
                _tail = self._level(_gap_closer.flush())
                if _tail:
                    rtc_track.push_pcm32k(_tail)
                if _gap_closer.closed:
                    logger.info(
                        f"[gap-closer] closed {_gap_closer.closed} mid-word "
                        f"silence(s) in {tts_text[:24]!r}"
                    )
            # ...and the stabiliser's own sub-frame remainder, so every sentence ships
            # complete (it holds <20ms while aligning to its 20ms analysis frames).
            if speakable and self._stream_cfg is not None:
                _rem = self._level_flush()
                if _rem:
                    if rtc_track is not None:
                        rtc_track.push_pcm32k(_rem)
                    else:
                        await websocket_send(
                            json.dumps(
                                {
                                    "type": "audio-stream-chunk",
                                    "data": base64.b64encode(_rem).decode("ascii"),
                                }
                            )
                        )

            # INTER-SENTENCE PAUSE: synthesis runs ahead of playback, so without
            # this the next sentence starts 0ms after the previous vowel ("次の
            # 文を話し始めるのが若干早い"). A punctuation-scaled breath makes
            # the boundary read naturally. (Re-enabled AFTER the 促音 artifact
            # was fixed in TTSAudioTrack — while that gap bug existed, inserted
            # pauses were masked by it.) Tune live: GSV_SENT_PAUSE_SCALE.
            if rtc_track is not None and pcm_parts:
                # NOTE: comma pause was 0.12s — INSIDE the geminate percept band
                # (≤~150ms = っ, ~300ms = natural 間, established by the tap
                # analysis). The first sentence is comma-split by faster_first_
                # response, so every reply opened with 促音-band pauses WE
                # inserted ourselves. All boundaries now sit in the 間 band.
                _t_end = tts_text.rstrip()
                # Language-keyed pause profile (follows the character switch):
                #  ja (UNCHANGED tuning): 間 band, 0.28 comma / 0.30 default.
                #  en: English cadence wants a longer full-stop breath — 0.30s
                #      ran sentences together and the body motion (scheduled
                #      per sentence) read as lagging the voice.
                if self._stream_cfg.text_lang == "ja":
                    _pause_map, _pause_default = {"、": 0.28, ",": 0.28, "，": 0.28}, 0.30
                else:
                    _pause_map, _pause_default = {
                        ",": 0.22, "、": 0.22,
                        ".": 0.40, "?": 0.45, "!": 0.40,
                    }, 0.40
                _pause_s = _pause_map.get(_t_end[-1] if _t_end else "", _pause_default)
                _pause_s *= float(os.getenv("GSV_SENT_PAUSE_SCALE", "1.0"))
                if _pause_s > 0:
                    rtc_track.push_pcm32k(b"\x00\x00" * int(_pause_s * PCM_SAMPLE_RATE))

            # Audio-driven co-speech gesture — PACED: one clip per
            # _GESTURE_MIN_SPACING_S of speech (plus one at the very start), not per
            # sentence, so GestureLSM keeps pace (its ~6s clip loops/crossfades to
            # cover the gap). Accumulate every sentence's spoken seconds regardless of
            # whether it fires, so short sentences still count toward the next clip.
            sent_secs = (
                len(b"".join(pcm_parts)) / 2 / PCM_SAMPLE_RATE if pcm_parts else 0.0
            )
            self._stream_total_secs += sent_secs
            logger.debug(
                f"[gesture-ledger] sentence done: +{sent_secs:.1f}s "
                f"(turn total {self._stream_total_secs:.1f}s)"
            )
            # After an early fire only the REMAINDER of the sentence counts
            # toward the next clip's spacing window. (Audio path only — the text
            # path tracks its own UNCOVERED-speech accumulator at fire time; a
            # fired sentence's own seconds are already covered by its clip.)
            if not text_gesture:
                self._stream_gesture_accum += max(0.0, sent_secs - early_fired_secs)
            gesture_due = (not self._stream_gesture_fired) or (
                self._stream_gesture_accum >= _GESTURE_MIN_SPACING_S
            )
            if (
                speakable
                and early_fired_secs == 0.0  # one clip per sentence (D2 fired already)
                and not text_gesture  # ActionPlan already fired from text
                and actions is not None
                and gesture_client.enabled()
                and pcm_parts
                and has_content
                and gesture_due
            ):
                try:
                    mid = motion_cache.start_gesture(
                        _pcm_to_wav_bytes(b"".join(pcm_parts)),
                        tts_text,
                        dur_override=sent_secs + 1.5,
                    )
                    actions.motion = {
                        "id": mid,
                        "key": "gesture",
                        "duration": estimate_speech_duration(tts_text),
                    }
                    self._stream_gesture_fired = True
                    self._stream_gesture_accum = 0.0
                    logger.info(
                        "[gesture-ledger] FIRE(sentence-end) at "
                        f"~{self._stream_total_secs:.1f}s of turn speech "
                        f"(window {sent_secs:.1f}s)"
                    )
                except Exception as _ge:  # noqa: BLE001
                    logger.warning(f"[gesture] stream fire failed: {_ge}")
            # HYBRID: KAWAII already fired the instant talk-loop from text
            # (text_gesture). For a LONG, non-emote sentence, ALSO generate GestureLSM
            # co-speech from the full audio and hand the frontend an upgrade_id to
            # crossfade over the loop when it lands. Short/emote sentences stay KAWAII.
            elif (
                text_gesture
                and not _upgrade_fired  # early-fire already handled it (the common path)
                and actions is not None
                and actions.motion
                and pcm_parts
                and has_content
                and _wants_gesturelsm_upgrade(actions, tts_text, sent_secs)
            ):
                try:
                    _gid = motion_cache.start_gesture(
                        _pcm_to_wav_bytes(b"".join(pcm_parts)),
                        tts_text,
                        dur_override=max(sent_secs + 1.5, 5.0),  # longest clip on offer
                    )
                    actions.motion["upgrade_id"] = _gid
                    self._stream_upgrade_n += 1
                    logger.info(
                        f"[gesture-ledger] HYBRID upgrade(GestureLSM,end) {_gid[:8]} "
                        f"over KAWAII for {sent_secs:.1f}s sentence"
                    )
                except Exception as _ge:  # noqa: BLE001
                    logger.warning(f"[gesture] hybrid upgrade fire failed: {_ge}")

            await websocket_send(
                json.dumps(
                    {
                        "type": "audio-stream-end",
                        "actions": actions.to_dict() if actions else None,
                        # This sentence's spoken duration — the WebRTC frontend
                        # uses it (with the start frame's play_in) to clock the
                        # co-speech gesture without a local audio buffer.
                        "audio_secs": round(sent_secs, 3),
                    }
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"[tts_manager] stream error for {tts_text[:30]!r}: {e}")
            try:
                await websocket_send(
                    json.dumps({"type": "audio-stream-end", "actions": None})
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            if not my_done.done():
                my_done.set_result(None)

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text (serialized via the global TTS lock)."""
        logger.debug(f"🏃Generating audio for '''{text}'''...")
        # Only one synthesis at a time so each request's timeout covers its own
        # work, not time spent waiting behind other sentences in the backend.
        async with _tts_synthesis_lock:
            return await tts_engine.async_generate_audio(
                text=text,
                file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            )

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        if self.streaming:
            # Cancel in-flight streams so an interrupted sentence stops sending
            # chunks immediately (each task's `finally` resolves its chain future).
            for t in self.task_list:
                if not t.done():
                    t.cancel()
            # Drop the chain tail so the next turn starts a fresh send order.
            self._stream_prev_done = None
            self._stream_gesture_accum = 0.0
            self._stream_gesture_fired = False
            self._stream_upgrade_n = 0
            self._stream_total_secs = 0.0
        self.task_list.clear()
        # Drop any in-flight bespoke motion generation for this abandoned turn.
        # (Safe in streaming now too: the frontend reports playback-complete only
        # when its player drains, so this cleanup runs AFTER the audio finished and
        # the turn's gestures were already generated + fetched — cancelling only
        # cancels stragglers that missed their window, freeing GestureLSM for the
        # next turn instead of letting clips pile up across turns.)
        motion_cache.cancel_all()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
