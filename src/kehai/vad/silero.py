import asyncio
from collections import deque
from enum import Enum

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from silero_vad import load_silero_vad

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
            required_misses=required_misses,
            smoothing_window=smoothing_window,
            end_misses=end_misses or 12,
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
        self.model = self.load_vad_model()
        self.state = StateMachine(self.config)
        self.window_size_samples = 512 if self.config.target_sr == 16000 else 256
        # 512 / 16000 = 0.032s

    def load_vad_model(self):
        logger.info("Loading Silero-VAD model...")
        return load_silero_vad()

    def detect_speech(self, audio_data: list[float]):
        audio_np = np.array(audio_data, dtype=np.float32)
        for i in range(0, len(audio_np), self.window_size_samples):
            chunk_np = audio_np[i : i + self.window_size_samples]
            if len(chunk_np) < self.window_size_samples:
                break
            chunk = torch.Tensor(chunk_np)

            with torch.no_grad():
                speech_prob = self.model(chunk, self.config.target_sr).item()

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
            return True
        finite = [d for d in self.dbs if np.isfinite(d)]
        if not finite:
            return False
        mean_db = float(np.mean(finite))
        if mean_db >= self.segment_mean_db:
            return True
        logger.info(
            f"[vad-level] {tag} DROPPED: mean {mean_db:.1f}dB < {self.segment_mean_db}dB "
            f"(too quiet to be speech aimed at the mic)"
        )
        return False

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

        # Obtain the smoothed prob and db
        smoothed_prob, smoothed_db = self.get_smoothed_values(prob, db)

        if self.state == State.IDLE:
            self.pre_buffer.append(chunk_bytes)
            # ONSET uses the strict proximity gate: this is where a far-away
            # voice or a room noise would otherwise open a turn.
            if smoothed_prob >= self.prob_threshold and smoothed_db >= self.onset_db:
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self.update(chunk_bytes, smoothed_prob, smoothed_db)
                    self.hit_count = 0
                    yield [], [], b"<|PAUSE|>"
            else:
                self.hit_count = 0

        elif self.state == State.ACTIVE:
            self.update(chunk_bytes, smoothed_prob, smoothed_db)
            if (
                smoothed_prob >= self.prob_threshold
                and smoothed_db >= self.db_threshold
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
                and smoothed_db >= self.db_threshold
            ):
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self.hit_count = 0
                    self.miss_count = 0
            else:
                self.hit_count = 0
                self.miss_count += 1
                if self.miss_count >= self.end_misses:
                    self.state = State.IDLE
                    self.miss_count = 0
                    yield [], [], b"<|RESUME|>"
                    # Any speech that accumulated since the last segment (i.e. a
                    # brief resume that didn't survive) goes out as a final
                    # segment; pure-silence tails are dropped by the prob guard.
                    if (
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
