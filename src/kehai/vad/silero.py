import copy
import os
import time
import asyncio
from collections import deque
from enum import Enum

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from silero_vad import load_silero_vad

from . import mic_tuning
from .. import turn_contingency
from .vad_interface import VADInterface


class SileroVADConfig(BaseModel):
    orig_sr: int = 16000
    target_sr: int = 16000
    prob_threshold: float = 0.4
    db_threshold: int = 60
    required_hits: int = 3  # 3 * (0.032) = 0.1s
    required_misses: int = 24  # 24 * (0.032) = 0.8s -> pause: closes a SEGMENT
    smoothing_window: int = 5
    # Additional misses in INACTIVE before the turn ends (IDLE). Total end-of-turn
    # silence = (required_misses + end_misses) * 0.032s. Kept separate so segment
    # pauses can be short (fast per-segment ASR) while turn-end stays tolerant.
    end_misses: int = 12  # 12 * 0.032 = ~0.4s
    # PROXIMITY GATE (2026-07-27): dB (int16 RMS scale) required to START
    # speech. Loudness is the only distance cue we have on one mic, and it is a
    # good one: every doubling of distance costs ~6 dB, so a gate a few dB under
    # "user at ~80cm" reliably ignores people further back in the room.
    # Deliberately SEPARATE from db_threshold, which stays lower and only keeps
    # an already-started utterance alive — otherwise a strict gate would chop
    # the quiet tail of your own sentence. 0 = fall back to db_threshold.
    onset_db_threshold: int = 0
    # MEAN-LEVEL GATE (2026-07-29): a finished segment is thrown away unless its
    # MEAN level reaches this. The onset gate above is a peak test, and peak does
    # not separate speech from noise — measured over 62 real segments with people
    # talking in the room, EVERYTHING peaked in the same 69-77 dB band, the user
    # and a laptop speaker and a door alike.
    #
    # The mean does separate them, cleanly. Real speech aimed at the mic measured
    # 40-59 dB mean; transients (a keystroke, a cough, a chair) measured 0-30 dB
    # mean with one brief peak. That matters more than it sounds, because
    # faster-whisper does not return nothing for a transient — it HALLUCINATES,
    # and always the same way: "Thank you." / "Thank you very much." / "Bye."
    # Twenty-two of those 62 segments were that, one of them at 0.3 dB mean. Each
    # became a conversation turn, interrupted whatever she was saying, and went
    # into the history the LLM reads. 0 disables the gate.
    segment_mean_db: int = 0
    # Log every emitted segment's level (dur / mean / peak dB) so the gate can be
    # calibrated from REAL room audio instead of guesswork. VAD_LEVEL_LOG=0 off.
    level_log: bool = True


class VADEngine(VADInterface):
    def __init__(
        self,
        orig_sr: int = 16000,
        target_sr: int = 16000,
        prob_threshold: float = 0.4,
        db_threshold: int = 60,
        required_hits: int = 3,
        required_misses: int = 24,
        smoothing_window: int = 5,
        end_misses: int = 12,
        onset_db_threshold: int = 0,
        segment_mean_db: int = 0,
    ):
        import os

        self.config = SileroVADConfig(
            orig_sr=orig_sr,
            target_sr=target_sr,
            prob_threshold=prob_threshold,
            db_threshold=db_threshold,
            required_hits=required_hits,
            # The PAUSE (segment close = where the speculative reply starts),
            # in 32 ms frames. Live override like VAD_END_MISSES: omni runs 4
            # (128 ms; conf.yaml's 6 = 192 ms) — the Thinker's first sentence is
            # the floor now, so every frame here moves the whole chain earlier.
            required_misses=int(os.getenv("VAD_PAUSE_MISSES", "") or required_misses or 24),
            smoothing_window=smoothing_window,
            # Live override too (start_server.sh sets it per brain mode: the omni
            # mode starts its speculative reply on the pause itself, so it can
            # afford a shorter end-of-turn wait than the cascade).
            end_misses=int(os.getenv("VAD_END_MISSES", "") or end_misses or 12),
            # Env override so the gate can be re-tuned live (calibration runs)
            # without editing conf.yaml and restarting twice.
            onset_db_threshold=int(
                os.getenv("VAD_ONSET_DB", "") or onset_db_threshold or 0
            ),
            # Same live-override treatment as the onset gate, and for the same
            # reason: this has to be tuned against the real room, not guessed.
            segment_mean_db=int(
                os.getenv("VAD_SEG_MEAN_DB", "") or segment_mean_db or 0
            ),
            level_log=os.getenv("VAD_LEVEL_LOG", "1") == "1",
        )
        if self.config.onset_db_threshold or self.config.segment_mean_db:
            logger.info(
                f"[vad] proximity gate: onset >= {self.config.onset_db_threshold}dB, "
                f"sustain >= {db_threshold}dB, "
                f"segment mean >= {self.config.segment_mean_db}dB"
            )
        # conf.yaml is the fallback for anything the mic panel never touched.
        mic_tuning.set_vad_defaults(
            prob_threshold=self.config.prob_threshold,
            onset_db=self.config.onset_db_threshold or self.config.db_threshold,
            sustain_db=self.config.db_threshold,
            segment_mean_db=self.config.segment_mean_db,
            onset_ms=self.config.required_hits * 32,
        )
        self.model = self.load_vad_model()
        self.state = StateMachine(self.config)
        self.window_size_samples = 512 if self.config.target_sr == 16000 else 256
        # 512 / 16000 = 0.032s

    def load_vad_model(self):
        logger.info("Loading Silero-VAD model...")
        return load_silero_vad()

    def fresh(self) -> "VADEngine":
        """Same settings, its own model and turn state — one per client.

        Sessions used to share the default context's engine: one state machine
        and one Silero RNN state for every connection, never reset. A reload
        mid-utterance carried the old bytes into the new client's first segment,
        and two clients streaming at once (the PROTO and a browser) fed one
        turn between them."""
        clone = copy.copy(self)
        clone.model = self.load_vad_model()
        clone.state = StateMachine(self.config)
        return clone

    _STATE_RESET_S = float(os.getenv("VAD_STATE_RESET_S", "2") or 0)

    def detect_speech(self, audio_data: list[float]):
        audio_np = np.array(audio_data, dtype=np.float32)
        for i in range(0, len(audio_np), self.window_size_samples):
            chunk_np = audio_np[i : i + self.window_size_samples]
            if len(chunk_np) < self.window_size_samples:
                break
            chunk = torch.Tensor(chunk_np)

            with torch.no_grad():
                speech_prob = self.model(chunk, self.config.target_sr).item()

            # The detector carries its memory from frame to frame and ran for
            # hours without ever clearing it. After a long quiet stretch, one
            # utterance and ~10 s of digital silence, a weak recording scored
            # 0.16-0.34 against the 0.4 it needs and left no trace (2026-11-09
            # 20:18-20:24, reproduced offline; 0.99 with the memory cleared).
            # Cleared after _STATE_RESET_S of clear silence while no turn is open.
            if self._STATE_RESET_S:
                idle = self.state.state == State.IDLE
                self._quiet_n = getattr(self, "_quiet_n", 0) + 1 if (idle and speech_prob < 0.1) else 0
                if self._quiet_n == int(self._STATE_RESET_S / 0.032):
                    self.model.reset_states()

            if speech_prob:
                # print(speech_prob)
                iter = self.state.get_result(speech_prob, chunk_np)

                for probs, dbs, chunk in iter:  # detected a sequence of voice bytes
                    # rounded_probs = [round(x, 2) for x in probs]
                    # rounded_dbs = [round(y, 2) for y in dbs]

                    audio_chunk = bytes(chunk)
                    yield audio_chunk

        del audio_np


# Define state enumeration
class State(Enum):
    IDLE = 1  # Idle state, waiting for speech
    ACTIVE = 2  # Speech detection state
    INACTIVE = 3  # Speech end state (silence state)


class StateMachine:
    def __init__(self, config: SileroVADConfig):
        self.state = State.IDLE
        self.prob_threshold = config.prob_threshold
        self.db_threshold = config.db_threshold
        # Proximity gate for STARTING speech (see SileroVADConfig). Falls back to
        # db_threshold, i.e. the original single-threshold behaviour.
        self.onset_db = config.onset_db_threshold or config.db_threshold
        self.segment_mean_db = config.segment_mean_db
        self.level_log = config.level_log
        self.required_hits = config.required_hits
        self.required_misses = config.required_misses
        self.end_misses = config.end_misses
        self.smoothing_window = config.smoothing_window

        self.probs = []
        self.dbs = []
        self.bytes = bytearray()
        self.miss_count = 0
        self.hit_count = 0

        self.prob_window = deque(maxlen=self.smoothing_window)
        self.db_window = deque(maxlen=self.smoothing_window)

        self.pre_buffer = deque(maxlen=20)
        self._tuning_version = -1

    def _apply_tuning(self):
        """Pick up the mic panel's thresholds (see mic_tuning) — checked every frame,
        applied only when they changed. required_hits is the impulse filter: a dropped
        object is loud and over in ~0.1 s, a word lasts longer. It only delays the
        moment a turn is RECOGNISED as started (the pre-buffer keeps the audio), not
        the end of the turn, so raising it costs no response time."""
        if self._tuning_version == mic_tuning.VERSION:
            return
        self._tuning_version = mic_tuning.VERSION
        v = mic_tuning.effective()
        self.prob_threshold = v.get("prob_threshold", self.prob_threshold)
        self.db_threshold = v.get("sustain_db", self.db_threshold)
        self.onset_db = v.get("onset_db", self.onset_db)
        self.segment_mean_db = v.get("segment_mean_db", self.segment_mean_db)
        if "onset_ms" in v:
            self.required_hits = max(1, round(v["onset_ms"] / 32))

    @classmethod
    def calculate_db(cls, audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf

    def update(self, chunk_bytes, prob, db):
        self.probs.append(prob)
        self.dbs.append(db)
        self.bytes.extend(chunk_bytes)

    def reset_buffers(self):
        self.probs.clear()
        self.dbs.clear()
        self.bytes.clear()

    def _log_level(self, tag: str) -> None:
        """Per-segment level report — the calibration data for the proximity
        gate. Read these while speaking at your normal seat, then again with
        someone talking from across the room: the two peak-dB clusters are what
        onset_db_threshold has to separate."""
        if not self.level_log or not self.dbs:
            return
        finite = [d for d in self.dbs if np.isfinite(d)]
        if not finite:
            return
        logger.info(
            f"[vad-level] {tag} dur={len(self.dbs) * 0.032:.2f}s "
            f"peak={max(finite):.1f}dB mean={float(np.mean(finite)):.1f}dB "
            f"(onset_gate={self.onset_db} sustain={self.db_threshold})"
        )

    def _loud_enough(self, tag: str) -> bool:
        """Log the segment's level, then say whether it should be transcribed.

        Both jobs together, because they read the same numbers and the log is the
        only place the decision is visible: a dropped segment leaves no other
        trace, and "she ignored me" would be indistinguishable from a bug.
        """
        self._log_level(tag)
        if not self.segment_mean_db:
            mic_tuning.MONITOR.event("pass", dur=round(len(self.dbs) * 0.032, 2))
            return True
        finite = [d for d in self.dbs if np.isfinite(d)]
        if not finite:
            return False
        mean_db = float(np.mean(finite))
        if mean_db >= self.segment_mean_db:
            mic_tuning.MONITOR.event("pass", mean=round(mean_db, 1), dur=round(len(self.dbs) * 0.032, 2))
            return True
        mic_tuning.MONITOR.event("drop", mean=round(mean_db, 1), dur=round(len(self.dbs) * 0.032, 2))
        logger.info(
            f"[vad-level] {tag} DROPPED: mean {mean_db:.1f}dB < {self.segment_mean_db}dB "
            f"(too quiet to be speech aimed at the mic)"
        )
        return False

    # The onset gate follows the room (2026-11-09, for a crowded venue): a fixed
    # 68 dB let the murmur of a full hall open turns on its own. The floor is the
    # 20th percentile of the last ~8 s of level heard while nobody was mid-turn;
    # the gate is that + VAD_ADAPT_MARGIN_DB (10), never below the configured gate
    # and never above VAD_ADAPT_MAX_DB (74 — the visitors' speech peaks at 70-80
    # dB with the PROTO's AGC). VAD_ADAPT_MARGIN_DB=0 turns it off.
    _ADAPT_MARGIN = float(os.getenv("VAD_ADAPT_MARGIN_DB", "10") or 0)
    _ADAPT_MAX = float(os.getenv("VAD_ADAPT_MAX_DB", "74") or 74)

    # Follow the speaker who opened the turn (2026-11-09 04:02): once a turn was
    # open, any speech at or above the sustain level (45 dB) kept it open, so with
    # someone in front of her the TV beside the PROTO ran on as "their" turn and
    # was answered. The turn's own level is taken from its first ~0.6 s of speech
    # (75th percentile); speech more than _REL_DROP dB under it (a voice further
    # away) neither keeps the turn open nor resumes it. Generous, because people
    # do trail off — the second half of a sentence comes back softer (see the
    # INACTIVE note below). VAD_REL_DROP_DB=0 turns it off.
    _REL_DROP = float(os.getenv("VAD_REL_DROP_DB", "15") or 0)

    def _sustain(self, smoothed_prob: float, smoothed_db: float) -> float:
        """The level that keeps this turn going (and learns the turn's level)."""
        ref_hist = getattr(self, "_ref_hist", None)
        if ref_hist is None:
            ref_hist = self._ref_hist = []
            self._ref, self._rel_cut = None, 0
        if self._ref is None and smoothed_prob >= self.prob_threshold and np.isfinite(smoothed_db):
            ref_hist.append(float(smoothed_db))
            if len(ref_hist) >= 20:
                self._ref = float(np.percentile(ref_hist, 75))
        if not self._REL_DROP or self._ref is None:
            return self.db_threshold
        sustain = max(self.db_threshold, self._ref - self._REL_DROP)
        if smoothed_prob >= self.prob_threshold and self.db_threshold <= smoothed_db < sustain:
            self._rel_cut += 1
        return sustain

    def _turn_reset(self) -> None:
        if getattr(self, "_rel_cut", 0) >= 10:
            logger.info(
                f"[vad] {self._rel_cut * 32} ms of speech more than {self._REL_DROP:.0f} dB under the "
                f"turn's own level ({self._ref:.0f} dB) not taken as the speaker's — a voice further away"
            )
        self._ref_hist, self._ref, self._rel_cut = [], None, 0

    def _base_gate(self) -> float:
        """The onset level before the noise floor is added: the mic panel's value,
        or — VAD_ROOM_GATE, see turn_contingency — just above the voices this
        room produces while nobody stands in front of her."""
        if not turn_contingency.ROOM_GATE:
            return self.onset_db
        gate, top, n = turn_contingency.room_gate()
        if abs(gate - getattr(self, "_room_gate_logged", -1.0)) >= 1.0:
            self._room_gate_logged = gate
            heard = "no voices heard with nobody in front" if top is None else f"room voices reach {top:.1f} dB over {n * 0.032:.0f} s"
            logger.info(f"[vad] room gate: onset {gate:.1f} dB ({heard}; mic panel {self.onset_db})")
        return gate

    def _adaptive_gate(self, smoothed_db: float) -> float:
        hist = getattr(self, "_floor_hist", None)
        if hist is None:
            hist = self._floor_hist = deque(maxlen=250)  # 8 s of 32 ms frames
            self._floor_gate, self._floor_n, self._floor_logged = self._base_gate(), 0, 0.0
        if np.isfinite(smoothed_db):
            hist.append(float(smoothed_db))
        self._floor_n += 1
        if self._floor_n % 31 == 0:  # ~1 s
            base = self._base_gate()
            self._floor_gate = base
            if self._ADAPT_MARGIN and len(hist) >= 60:
                floor = float(np.percentile(hist, 20))
                self._floor_gate = min(max(base, floor + self._ADAPT_MARGIN), max(base, self._ADAPT_MAX))
                now = time.monotonic()
                if self._floor_gate > base and now - self._floor_logged > 30:
                    self._floor_logged = now
                    logger.info(f"[vad] noisy room: onset gate {self._floor_gate:.0f} dB (floor {floor:.0f} dB)")
        return self._floor_gate

    def get_smoothed_values(self, prob, db):
        self.prob_window.append(prob)
        self.db_window.append(db)
        smoothed_prob = np.mean(self.prob_window)
        smoothed_db = np.mean(self.db_window)
        return smoothed_prob, smoothed_db

    def process(self, prob, float_chunk_np: np.ndarray):
        int_chunk_np = float_chunk_np * 32767
        chunk_bytes = int_chunk_np.astype(np.int16).tobytes()
        db = self.calculate_db(int_chunk_np)

        self._apply_tuning()
        # Obtain the smoothed prob and db
        smoothed_prob, smoothed_db = self.get_smoothed_values(prob, db)
        mic_tuning.MONITOR.frame(smoothed_prob, smoothed_db, self.state.name.lower())
        turn_contingency.frame(smoothed_prob, smoothed_db, self.prob_threshold, self.db_threshold)

        if self.state == State.IDLE:
            self.pre_buffer.append(chunk_bytes)
            gate = self._adaptive_gate(smoothed_db)
            # ONSET uses the strict proximity gate: this is where a far-away
            # voice or a room noise would otherwise open a turn.
            if smoothed_prob >= self.prob_threshold and smoothed_db >= gate:
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self._turn_reset()
                    self.update(chunk_bytes, smoothed_prob, smoothed_db)
                    self.hit_count = 0
                    self._under = 0
                    mic_tuning.MONITOR.event("start", db=round(float(smoothed_db), 1))
                    yield [], [], b"<|PAUSE|>"
            else:
                self.hit_count = 0
                # Speech the onset gate kept out leaves no trace otherwise, and
                # "sometimes she does not answer" could not be told apart from
                # the addressee checks (2026-11-04). Log a run of clear speech
                # (Silero sure) that stayed under the gate for >= 0.32 s.
                if smoothed_prob >= max(self.prob_threshold, 0.6):
                    self._under = getattr(self, "_under", 0) + 1
                    self._under_db = max(getattr(self, "_under_db", -1e9), float(smoothed_db))
                elif getattr(self, "_under", 0):
                    if self._under >= 10:
                        logger.info(
                            f"[vad-level] speech UNDER the onset gate: {self._under * 32} ms, "
                            f"loudest {self._under_db:.1f} dB < {gate:.0f} dB — not a turn"
                        )
                    self._under, self._under_db = 0, -1e9

        elif self.state == State.ACTIVE:
            self.update(chunk_bytes, smoothed_prob, smoothed_db)
            if (
                smoothed_prob >= self.prob_threshold
                and smoothed_db >= self._sustain(smoothed_prob, smoothed_db)
            ):
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count >= self.required_misses:
                    self.state = State.INACTIVE
                    self.miss_count = 0
                    # A pause closes the current speech SEGMENT — emit it now so
                    # ASR can run on it while the user keeps talking (or while
                    # the end-of-turn silence is still being waited out).
                    if (
                        len(self.probs) > 8
                        and max(self.probs, default=0.0) >= self.prob_threshold
                        and self._loud_enough("SEG")
                    ):
                        pre_bytes = b"".join(self.pre_buffer)
                        yield (
                            self.probs,
                            self.dbs,
                            b"<|SEG|>" + pre_bytes + bytes(self.bytes),
                        )
                    self.reset_buffers()
                    self.pre_buffer.clear()

        elif self.state == State.INACTIVE:
            self.update(chunk_bytes, smoothed_prob, smoothed_db)
            # RESUMING your own turn uses the lenient SUSTAIN level, not the
            # proximity gate. Gating the resume at onset level chopped sentences
            # in half: you open loudly, take a breath, and the second half comes
            # back softer than the gate, so the turn ended and the ASR only ever
            # saw fragments ("what time it is done"). Distance filtering happens
            # at turn START (IDLE); once the turn is yours, keep it.
            if (
                smoothed_prob >= self.prob_threshold
                and smoothed_db >= self._sustain(smoothed_prob, smoothed_db)
            ):
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self.hit_count = 0
                    self.miss_count = 0
                    # The user picked the turn back up after a segment closed.
                    # Nothing else marks this (<|RESUME|> is the turn END), and a
                    # reply started speculatively on the pause must be dropped.
                    yield [], [], b"<|SPEECH|>"
            else:
                self.hit_count = 0
                self.miss_count += 1
                if self.miss_count >= self.end_misses:
                    self.state = State.IDLE
                    self.miss_count = 0
                    self._turn_reset()
                    yield [], [], b"<|RESUME|>"
                    # Any speech that accumulated since the last segment (i.e. a
                    # brief resume that didn't survive) goes out as a final
                    # segment; pure-silence tails are dropped by the prob guard.
                    # ...and only if it ever reached the SUSTAIN level, the one
                    # that keeps speech alive in ACTIVE. Tails peaking at 39-44 dB
                    # (breath, a chair, a far voice) passed the other checks, and
                    # with cumulative ASR each one re-transcribed the WHOLE
                    # utterance at the endpoint: 1.2-1.3 s of extra wait, a
                    # transcript that changed under the speculative reply, and
                    # that reply thrown away (2026-10-23 21:55:37, 22:19:28).
                    tail_peak = max(
                        (d for d in self.dbs if np.isfinite(d)), default=-np.inf
                    )
                    if (
                        len(self.probs) > 8
                        and max(self.probs, default=0.0) >= self.prob_threshold
                        and tail_peak < self.db_threshold
                    ):
                        self._log_level("SEG-END below sustain, not transcribed")
                    elif (
                        len(self.probs) > 8
                        and max(self.probs, default=0.0) >= self.prob_threshold
                        and self._loud_enough("SEG-END")
                    ):
                        pre_bytes = b"".join(self.pre_buffer)
                        yield (
                            self.probs,
                            self.dbs,
                            b"<|SEG|>" + pre_bytes + bytes(self.bytes),
                        )
                    self.reset_buffers()
                    self.pre_buffer.clear()
                    # Turn boundary: the handler finalizes the accumulated
                    # segment transcripts into one conversation trigger.
                    yield [], [], b"<|TURN_END|>"

    def get_result(self, input_num, chunk_np):
        yield from self.process(input_num, chunk_np)


async def vad_main():
    global vad, audio_queue
    vad = VADEngine(config=SileroVADConfig())
    audio_queue = asyncio.Queue()
    from tqdm.asyncio import tqdm

    async def data_wrapper(websocket):
        async for chunk in websocket:
            yield chunk

    async def audio_handler(websocket):
        async for chunk in tqdm(data_wrapper(websocket), desc="Audio chunk"):
            # print(len(chunk))
            for _bytes in vad.detect_speech(chunk):
                print(_bytes[:44])
                # await audio_queue.put(_bytes)
                pass

    async def empty_run():
        while True:
            await asyncio.sleep(0.1)

    async def start_websocket_server():
        import websockets

        host = "localhost"
        port = 8765
        start_server = websockets.serve(audio_handler, host, port)
        logger.info(f"WebSocket server started at ws://{host}:{port}")
        await start_server  # run forever until the task is cancelled

    await asyncio.gather(start_websocket_server(), empty_run())
    # await start_playback(audio_queue, sr=vad.config.target_sr)


if __name__ == "__main__":
    asyncio.run(vad_main())
