"""Opus encoding for the WEBSOCKET audio path.

Why this exists
---------------
When a WebRTC peer is up, her voice rides an Opus track and costs ~32-64 kbps.
When it is not — which is the normal case on a box reached through an SSH port
forward, where ICE can never complete because aiortc only offers the server's
own host candidates — `tts_manager` falls back to sending the raw synth output
over the WebSocket instead:

    32000 Hz x 16 bit mono = 512 kbps, and base64 in a JSON envelope = 683 kbps

That is the SAME speech, ~15x larger, on the path that is by definition the
constrained one. Measured against a VS Code port forward (cwnd 10 at 64 ms RTT
=> ~1.7 Mbps of usable window, shared with everything else VS Code does) the
raw stream alone wanted ~40% of the pipe, so any competing transfer stalled the
audio behind it. Encoding it here brings the fallback in line with the fast
path and drops that to a few percent.

The encoder is deliberately built here rather than reusing
`aiortc.codecs.opus.OpusEncoder`: that one is hard-wired to STEREO at 96 kbps
because it serves an RTP track. Mono at 64 kbps is half the bytes for the same
audio, and mono is what the synth actually produces.

Base64 is kept. At ~64 kbps the 33% inflation is ~21 kbps — cheaper than
introducing binary WebSocket frames, which would mean a tag-byte protocol, a
`binaryType` change, and reasoning about ordering between text and binary
frames for the sake of a stream that is already small.
"""

from __future__ import annotations

import fractions
import os

from av import AudioResampler
from av.audio.frame import AudioFrame
from av.codec import CodecContext
from loguru import logger

# Opus only accepts 8/12/16/24/48 kHz. The synth gives 32 kHz, so everything is
# resampled to 48 kHz — the same conversion the WebRTC track already does.
OPUS_RATE = 48000
# 20 ms at 48 kHz. libopus encodes whole frames; the resampler is told to emit
# exactly this many samples so the encoder never has to buffer a partial one.
SAMPLES_PER_FRAME = 960
PCM_IN_RATE = 32000


# Which clients told us they can decode Opus, keyed by id() of their WebSocket.
# Opus decoding in the browser needs WebCodecs (`AudioDecoder`), which not every
# engine ships, so this is NEGOTIATED rather than assumed: a client that stays
# quiet keeps getting raw PCM and nothing breaks. Mirrors the `_by_ws` lookup in
# webrtc_audio.py — the TTS loop only ever holds a bound `websocket.send_text`,
# so the socket is recovered from the bound method's `__self__`.
_opus_clients: set[int] = set()


def set_client_opus(websocket, supported: bool) -> None:
    if supported:
        _opus_clients.add(id(websocket))
    else:
        _opus_clients.discard(id(websocket))


def forget_client(websocket) -> None:
    _opus_clients.discard(id(websocket))


def client_supports_opus(websocket_send) -> bool:
    ws = getattr(websocket_send, "__self__", None)
    if ws is not None and getattr(ws, "pcm_only", False):
        return False  # render_stream.ViewerMirror decodes PCM for the viewer
    return ws is not None and id(ws) in _opus_clients


def opus_bitrate() -> int:
    """Bits per second for the WS Opus stream. `WS_OPUS_BITRATE=0` disables Opus."""
    try:
        return int(os.getenv("WS_OPUS_BITRATE", "64000"))
    except ValueError:
        return 64000


def opus_enabled() -> bool:
    return opus_bitrate() > 0


class WsOpusEncoder:
    """Raw 32 kHz s16 mono PCM in, Opus packets out. One instance per turn.

    `feed()` accepts chunks of any size — the synth yields whatever the model
    produced, including odd byte counts — and returns only the packets that are
    complete. Call `flush()` at the end of the turn for the tail.
    """

    def __init__(self, bitrate: int | None = None) -> None:
        rate = bitrate if bitrate is not None else opus_bitrate()
        self._codec = CodecContext.create("libopus", "w")
        self._codec.bit_rate = rate
        self._codec.format = "s16"
        self._codec.layout = "mono"
        # "audio", not aiortc's "voip": voip applies speech-band shaping that
        # dulls the sibilance and upper harmonics this voice is largely made of.
        # At 64 kbps there is no bit-budget reason to take that trade.
        self._codec.options = {
            "application": os.getenv("WS_OPUS_APPLICATION", "audio"),
            # CONSTRAINED, not libopus' default unconstrained VBR. The whole
            # point of this path is a predictable ceiling on a pipe that is
            # already contended; unconstrained VBR overshot the target by ~65%
            # on dense material in testing, which is exactly the moment the link
            # can least afford it. Constrained VBR keeps the quality benefit of
            # VBR while holding the average to `bit_rate`.
            "vbr": os.getenv("WS_OPUS_VBR", "constrained"),
        }
        self._codec.sample_rate = OPUS_RATE
        self._codec.time_base = fractions.Fraction(1, OPUS_RATE)
        self._resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=OPUS_RATE,
            frame_size=SAMPLES_PER_FRAME,
        )
        # Odd trailing byte carried to the next chunk. Dropping it would shift
        # every following sample by one byte — continuous noise, not a click.
        self._inbuf = b""
        self._pts = 0
        self.bitrate = rate

    def feed(self, pcm: bytes) -> list[bytes]:
        if not pcm:
            return []
        data = self._inbuf + pcm
        usable = (len(data) // 2) * 2
        self._inbuf = data[usable:]
        samples = usable // 2
        if samples <= 0:
            return []
        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = PCM_IN_RATE
        frame.time_base = fractions.Fraction(1, PCM_IN_RATE)
        frame.pts = self._pts
        self._pts += samples
        frame.planes[0].update(data[:usable])
        return self._encode(frame)

    def flush(self) -> list[bytes]:
        """Drain the resampler and the encoder so the last words are not clipped."""
        out: list[bytes] = []
        try:
            for resampled in self._resampler.resample(None):
                out += [bytes(p) for p in self._codec.encode(resampled)]
            out += [bytes(p) for p in self._codec.encode(None)]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ws-opus] flush failed: {e!r}")
        return out

    def _encode(self, frame: AudioFrame) -> list[bytes]:
        out: list[bytes] = []
        try:
            for resampled in self._resampler.resample(frame):
                out += [bytes(p) for p in self._codec.encode(resampled)]
        except Exception as e:  # noqa: BLE001
            # Never let a codec hiccup take the voice down: the caller falls back
            # to sending PCM for the rest of the turn.
            logger.warning(f"[ws-opus] encode failed, dropping to PCM: {e!r}")
            raise
        return out
