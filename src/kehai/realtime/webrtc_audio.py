"""WebRTC audio transport (Phase G).

Swaps the two AUDIO paths onto one RTCPeerConnection while every control
message (signaling, subtitles, actions, motion ids, interrupts) stays on the
existing WebSocket:

  Up:   browser mic track (Opus 48k) -> resample 16k mono float32 ->
        512-sample chunks into the SAME server-VAD segment pipeline the WS
        ``raw-audio-data`` path uses (websocket_handler._process_vad_chunk).
  Down: tts_manager's streamed GSV PCM (32k s16 mono) -> resample 48k ->
        a custom outbound MediaStreamTrack the browser plays natively.
        The browser's jitter buffer replaces the hand-rolled preroll, base64
        framing disappears, and — critically — the browser AEC finally sees
        the AI's voice as its render reference, so speaker echo is cancelled
        properly instead of ducked around.

Signaling: the client sends {"type": "webrtc-offer", "sdp": ...} over the WS;
we answer with {"type": "webrtc-answer", "sdp": ...}. LAN host candidates are
sufficient (no STUN/TURN) because client and server share the network.
"""

import asyncio
import fractions
import os
import time
from typing import Awaitable, Callable, Dict, Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler
from loguru import logger

PCM_IN_RATE = 32000  # GSV streaming output (s16 mono)
RTC_RATE = 48000  # WebRTC canonical clock
VAD_RATE = 16000  # server VAD / ASR rate
VAD_CHUNK = 512  # samples per Silero window (32ms @16k)

_FRAME_SAMPLES = 960  # 20ms @48k
_FRAME_BYTES = _FRAME_SAMPLES * 2
# The keep-alive frame, kept as a constant so recv() can tell "silence I made up"
# apart from "speech I was given" without re-scanning the buffer.
_SILENT_FRAME = b"\x00" * _FRAME_BYTES

# The outbound-speech counter below is DIAGNOSTIC ONLY (WEBRTC_OUT_LOG=1).
#
# It answered the question it was added for — proving the server really put audio
# on the wire while a browser stayed silent — and then kept running. Its "is this
# speech?" test is an exact byte compare against a silent frame, which the
# post-gap fade-in and the level stabiliser defeat: a barely-audible frame is not
# byte-identical to silence, so a quiet track counted as continuous speech and
# logged every second without end (measured: 99 seconds straight, immediately
# before a disconnect). A permanent per-second log on the media path is the wrong
# default regardless of what that disconnect turns out to be.
_OUT_LOG = os.getenv("WEBRTC_OUT_LOG", "0") == "1"


class TTSAudioTrack(MediaStreamTrack):
    """Outbound AI-speech track: 20ms s16 frames pulled from a PCM queue that
    tts_manager feeds; emits silence whenever the queue is empty so the track
    never starves (and the AEC reference stays continuous)."""

    kind = "audio"

    # Lead built up before a speech burst starts playing. Without it the track
    # consumes the very first bytes immediately and every micro-stall in GSV's
    # bursty synthesis interleaves 20ms silence frames into the speech — heard
    # as continuous crackle. (The WS player solved the same problem with its
    # 0.5s preroll; 0.3s suffices here since RTF≈0.5 grows the lead quickly.)
    PRIME_SECONDS = 0.12  # was 0.15; shaved with the rest of the onset budget
    # Re-prime lead after a MID-TURN dry-out. Tap analysis (139s of real
    # playback) found the glottal-stop artifact ("句読点で小さい「っ」"): 29
    # supply gaps of 30-150ms at cut5 fragment boundaries — acoustically a
    # PERFECT Japanese geminate (closure + consonant burst). The fix is NOT a
    # shorter gap (50ms was peak-っ) but a LONGER one: ~300ms reads as a
    # natural reading pause (間) instead of a っ closure. Perception boundary,
    # not a buffering problem.
    REPRIME_SECONDS = 0.30
    # MINIMUM audible gap once a dry-out happened. The buffer threshold alone
    # can't shape the gap: fragments arrive as BURSTS, so playback would resume
    # after just the supply latency (the measured 30-90ms っ). Enforcing a
    # 0.3s floor turns every unavoidable gap into a natural reading pause.
    MIN_GAP_SECONDS = 0.30
    # Post-gap fade-in: soften the consonant burst right after a re-prime gap
    # (the crisp attack is what makes the closure register as 促音).
    FADE_IN_SECONDS = 0.02

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._resampler = AudioResampler(format="s16", layout="mono", rate=RTC_RATE)
        self._pts = 0
        self._t0: Optional[float] = None
        self._priming = True
        self._prime_target = self.PRIME_SECONDS
        self._dry_since: Optional[float] = None
        self._fade_in = 0  # samples left of the post-gap fade-in
        # Debug tap (WEBRTC_TAP=1): record the EXACT 48k s16 PCM this track
        # emits (speech + inserted silence) to cache/webrtc_tap.raw, so a
        # perceived playback artifact can be located objectively — in-source
        # (GSV) vs in-track (priming/joins) vs in-browser.
        self._tap = None
        if os.getenv("WEBRTC_TAP", "0") == "1":
            try:
                os.makedirs("cache", exist_ok=True)
                self._tap = open("cache/webrtc_tap.raw", "ab")
            except OSError:
                self._tap = None
        # Trailing byte of an odd-length chunk: GSV's HTTP chunks split int16
        # samples across (TCP) boundaries. Dropping the odd byte byte-shifts
        # every later sample = continuous harsh noise (the WS player fixed the
        # SAME bug with its `leftover` carry).
        self._inbuf = b""
        self._in_pts = 0  # monotonic input pts so the resampler never gap-fills
        # 20ms frames of real speech handed to the encoder (see recv()).
        self._speech_frames = 0

    def push_pcm32k(self, pcm: bytes) -> None:
        """Feed raw 32k s16 mono PCM (any chunk size; called from the TTS loop)."""
        if not pcm or self.readyState != "live":
            return
        # Reassemble int16 samples across arbitrary chunk boundaries: keep the
        # odd trailing byte for the next chunk instead of dropping it (dropping
        # shifts every subsequent sample by one byte -> continuous noise).
        data = self._inbuf + pcm
        usable = (len(data) // 2) * 2
        self._inbuf = data[usable:]
        samples = usable // 2
        if samples <= 0:
            return
        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = PCM_IN_RATE
        frame.pts = self._in_pts
        frame.time_base = fractions.Fraction(1, PCM_IN_RATE)
        self._in_pts += samples
        frame.planes[0].update(data[:usable])
        for out in self._resampler.resample(frame):
            self._buffer.extend(bytes(out.planes[0])[: out.samples * 2])

    def clear(self) -> None:
        """Drop any queued speech (barge-in / interrupt)."""
        self._buffer.clear()
        self._inbuf = b""
        self._priming = True
        self._prime_target = self.PRIME_SECONDS  # next turn: full initial lead

    def buffered_seconds(self) -> float:
        return len(self._buffer) / 2 / RTC_RATE

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        # Real-time pacing: emit exactly one 20ms frame per 20ms of wall clock.
        if self._t0 is None:
            self._t0 = time.monotonic()
        target = self._t0 + (self._pts / RTC_RATE)
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        if (
            self._priming
            and self.buffered_seconds() >= self._prime_target
            and (
                self._dry_since is None
                or time.monotonic() - self._dry_since >= self.MIN_GAP_SECONDS
            )
        ):
            self._priming = False
            if self._dry_since is not None:
                # resuming after a mid-turn gap: soften the first attack
                self._fade_in = int(self.FADE_IN_SECONDS * RTC_RATE)
                # attribution log: any future "っ" report can be checked against
                # these — no [track-gap] lines at the time => the closure was in
                # the GENERATED audio, not a playback gap.
                logger.info(
                    f"[track-gap] resumed after "
                    f"{(time.monotonic() - self._dry_since) * 1000:.0f}ms"
                )
            self._dry_since = None
        if not self._priming and len(self._buffer) >= _FRAME_BYTES:
            chunk = bytes(self._buffer[:_FRAME_BYTES])
            del self._buffer[:_FRAME_BYTES]
            if self._fade_in > 0:
                arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                total = int(self.FADE_IN_SECONDS * RTC_RATE)
                k = min(len(arr), self._fade_in)
                lo = 1.0 - self._fade_in / total
                hi = 1.0 - (self._fade_in - k) / total
                arr[:k] *= np.linspace(lo, hi, k, dtype=np.float32)
                self._fade_in -= k
                chunk = arr.astype(np.int16).tobytes()
        else:
            if not self._priming and len(self._buffer) < _FRAME_BYTES:
                # Ran dry mid-burst: rebuild the lead as ONE clean gap instead
                # of spraying 20ms silence slivers into the speech (crackle).
                # SHORT re-prime (50ms): a sentence-boundary shortfall resumes
                # almost immediately instead of stretching into a glottal-stop
                # sized hole. Drop the sub-frame leftover — replaying a ~10ms
                # scrap of the previous vowel after the gap reads as a pop.
                self._priming = True
                self._prime_target = self.REPRIME_SECONDS
                self._dry_since = time.monotonic()
                self._buffer.clear()
            elif self._priming and self._dry_since is not None:
                # Long dry spell = turn boundary, not a mid-turn hiccup:
                # restore the full initial prime for the next reply.
                if time.monotonic() - self._dry_since > 1.0:
                    self._prime_target = self.PRIME_SECONDS
                    self._dry_since = None
            chunk = _SILENT_FRAME  # silence keep-alive
        # How much SPEECH (not keep-alive silence) this track has actually handed
        # to the encoder. "No sound came out" is otherwise unanswerable from the
        # server side: the TTS log proves audio was synthesised and the gap-closer
        # log proves it reached this object, but neither proves a frame was ever
        # emitted. One line per second of speech settles it — if these appear,
        # the fault is downstream of the server, in the network or the browser.
        if _OUT_LOG and chunk != _SILENT_FRAME:
            self._speech_frames += 1
            if self._speech_frames % 50 == 0:  # 50 x 20ms = 1s of speech
                logger.info(
                    f"[webrtc-out] emitted {self._speech_frames * 20 / 1000:.0f}s "
                    "of speech to the peer"
                )
        if self._tap is not None:
            try:
                self._tap.write(chunk)
            except OSError:
                pass
        frame = AudioFrame(format="s16", layout="mono", samples=_FRAME_SAMPLES)
        frame.sample_rate = RTC_RATE
        frame.planes[0].update(chunk)
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RTC_RATE)
        self._pts += _FRAME_SAMPLES
        return frame


async def _consume_mic(
    track: MediaStreamTrack,
    on_chunk: Callable[[np.ndarray], Awaitable[None]],
) -> None:
    """Read the browser's mic track, resample to 16k mono float32 and hand the
    VAD 512-sample windows — the same shape the WS raw-audio-data path feeds."""
    resampler = AudioResampler(format="s16", layout="mono", rate=VAD_RATE)
    pending = np.zeros(0, dtype=np.float32)
    try:
        while True:
            frame = await track.recv()
            for out in resampler.resample(frame):
                pcm = np.frombuffer(
                    bytes(out.planes[0])[: out.samples * 2], dtype=np.int16
                )
                pending = np.concatenate([pending, pcm.astype(np.float32) / 32768.0])
                while len(pending) >= VAD_CHUNK:
                    await on_chunk(pending[:VAD_CHUNK])
                    pending = pending[VAD_CHUNK:]
    except MediaStreamError:
        logger.debug("[webrtc] mic track ended")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[webrtc] mic consumer stopped: {e!r}")


class WebRTCManager:
    """Per-client peer registry. Keyed by client_uid, with a WebSocket-object
    index so tts_manager (which only holds websocket_send) can find the track."""

    def __init__(self) -> None:
        self._peers: Dict[str, RTCPeerConnection] = {}
        self._tracks: Dict[str, TTSAudioTrack] = {}
        self._by_ws: Dict[int, str] = {}  # id(websocket) -> client_uid

    async def handle_offer(
        self,
        websocket,
        client_uid: str,
        sdp: str,
        on_mic_chunk: Callable[[np.ndarray], Awaitable[None]],
    ) -> str:
        """Accept the browser's offer; returns the answer SDP."""
        await self.close(client_uid)
        pc = RTCPeerConnection()
        tts_track = TTSAudioTrack()
        pc.addTrack(tts_track)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                logger.info(f"[webrtc] mic track connected for {client_uid}")
                asyncio.ensure_future(_consume_mic(track, on_mic_chunk))

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            logger.info(f"[webrtc] {client_uid} state={pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                await self.close(client_uid)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)  # aiortc gathers host candidates here
        self._peers[client_uid] = pc
        self._tracks[client_uid] = tts_track
        self._by_ws[id(websocket)] = client_uid
        logger.info(f"[webrtc] answered offer for {client_uid}")
        return pc.localDescription.sdp

    def track_for(self, client_uid: str) -> Optional[TTSAudioTrack]:
        return self._tracks.get(client_uid)

    def track_for_send(self, websocket_send) -> Optional[TTSAudioTrack]:
        """Resolve a track from the bound ``websocket.send_text`` the TTS layer
        holds (its __self__ is the WebSocket).

        Only a CONNECTED peer gets the audio. A track exists from the moment the
        offer is answered, but a peer still negotiating — or wedged in
        'connecting' forever because ICE can never complete (a tunnelled or
        reverse-proxied origin reaches none of our host candidates) — cannot
        deliver a single sample. The TTS layer branches on `track is not None`
        and sends over the WebSocket only in the else, so handing back a track
        for such a peer both DISCARDS the audio and SUPPRESSES the fallback:
        the avatar just goes mute. Measured on the ui-proto origin: the peer sat
        in 'connecting' for 63s before ICE gave up, and all three turns spoken
        in that window were silent while the server logged a normal TTFA.
        """
        ws = getattr(websocket_send, "__self__", None)
        if ws is None:
            return None
        uid = self._by_ws.get(id(ws))
        if not uid:
            return None
        pc = self._peers.get(uid)
        if pc is None or pc.connectionState != "connected":
            return None
        return self._tracks.get(uid)

    def clear_audio(self, client_uid: str) -> None:
        track = self._tracks.get(client_uid)
        if track:
            track.clear()

    async def close(self, client_uid: str) -> None:
        pc = self._peers.pop(client_uid, None)
        self._tracks.pop(client_uid, None)
        self._by_ws = {k: v for k, v in self._by_ws.items() if v != client_uid}
        if pc:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass


async def notify_when_drained(
    websocket_send, track: TTSAudioTrack, margin_s: float = 0.4, timeout_s: float = 90.0
) -> None:
    """Server-authoritative end-of-turn: after synthesis completes, watch the
    outbound track's buffer and tell the frontend the moment it has fully played
    out (+ a jitter margin). Replaces the frontend's analyser silence-watch,
    which misfired on quiet passages and on synth-complete-before-playback
    (killing gestures + talking motion mid-turn)."""
    import json as _json

    start = time.monotonic()
    try:
        while time.monotonic() - start < timeout_s:
            if track.readyState != "live":
                return
            if track.buffered_seconds() <= 0.02:
                await asyncio.sleep(margin_s)
                if track.buffered_seconds() <= 0.02:
                    await websocket_send(_json.dumps({"type": "playback-drained"}))
                    return
            else:
                await asyncio.sleep(0.1)
    except Exception:  # noqa: BLE001 — a lost drain signal only delays the failsafe
        pass


webrtc_manager = WebRTCManager()
