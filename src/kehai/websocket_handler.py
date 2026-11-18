from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import base64
import io
import json
import os
import re
import time
import wave
from collections import deque
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .realtime import webrtc_manager, ws_opus
from .utils.turn_timing import turn_timing
from .utils import asr_capture
from .render_stream import hub as render_hub
from .conversations import speculative
from . import face_detect
from . import voice_direction
from . import omni_addressee
from . import active_speaker
from . import speaker_probe
from . import turn_contingency
from . import lip_motion
from .vad import smart_turn
import functools
from scipy.signal import firwin, lfilter
from .conversations.conversation_handler import (
    apply_client_language,
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


# --- Semantic endpointing (server-VAD path) -------------------------------
# Grace period a mid-thought turn stays open for; resuming speech within it
# folds the held transcript into the next turn instead of firing early.
_VAD_GRACE_S = 0.8
# How long a turn whose words end mid-sentence ("Can you tell me about") waits
# for the speaker to go on. 0.8 s was too short on the PROTO: after a pause the
# onset gate (68 dB, smoothed) took ~0.9 s to call the second half speech, so
# the hold ran out and she answered the half (11-07 probe: pause 0.9 s, second
# half's onset 1.2 s into the hold).
_UNFINISHED_HOLD_S = float(os.getenv("VAD_UNFINISHED_HOLD_S", "1.5") or 1.5)
# A visitor who goes on talking within this long after the end of a turn she took
# (its <|TURN_END|>) is the same visitor continuing — the second half of a sentence
# split by a pause. Its speech is not judged again by who-is-talking: heard alone,
# "the South Market, please" is a fragment, and judged as a new utterance it was
# dropped or failed to stop her (2026-11-07). Timing only, no words: an onset this
# soon after her turn began is ~a 1.5-1.9 s pause in the speaker's own speech.
_CONT_S = float(os.getenv("TURN_CONTINUE_S", "1.5") or 1.5)

# asyncio keeps only a weak reference to a task: one nobody holds can be
# collected while it waits, and its turn would end without a word or a log line.
_TASKS: set = set()


def _spawn(coro) -> "asyncio.Task":
    t = asyncio.create_task(coro)
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
    return t

# Speculative replies (conversations/speculative.py): start the reply on the
# segment pause, publish it on the endpoint. SPECULATE=0 turns it off.
_SPECULATE = os.getenv("SPECULATE", "1") == "1"
# omni mode, audio-first speculation: start the reply on the PAUSE with the audio
# so far, before the segment ASR returns (~0.43 s). The transcript still screens
# the run (junk / mid-thought) and replaces the placeholder it stored; the HIT
# test at the endpoint stays textual. start_server.sh turns this on in omni mode.
_SPEC_AUDIO_FIRST = os.getenv("SPECULATE_AUDIO_FIRST", "0") == "1"
# FAST RELEASE (omni, audio-first): at the endpoint, a speculative reply that
# the user did not talk over is released at once, WITHOUT waiting for the ASR
# transcript (0.3-0.45 s after the pause on this box, i.e. 0.5-0.65 s after the
# last voice — the thing that put a floor under VAD_END_MISSES). The Thinker
# heard the audio itself; the words are only for the chat panel, the history
# and the junk screen. The screen runs when the transcript lands: a turn that
# proves to be noise ("Thank you." on a cough) is retracted — the reply is
# cancelled and rolled back (speculative.discard) — so she may start a syllable
# and stop. Utterances shorter than SPEC_FAST_RELEASE_MIN_S take the old path
# (short sounds are where the hallucinations come from). 2026-11-04.
_FAST_RELEASE = os.getenv("SPEC_FAST_RELEASE", "0") == "1"
# VAD_VIEWER_DIRECT (render client only): run the server VAD on the PROTO
# viewer's mic as it reaches the server (render_stream, 48 kHz kind-3 frames)
# instead of on the render tab's round-trip copy, which lags it 81-111 ms
# (viewer -> tab mic worklet -> MediaStream -> vad-web -> raw-audio-data -> here)
# before the first VAD frame exists. Every later stage (pause, speculation,
# endpoint) moves earlier by that much. The tab's copy is then ignored for
# VAD. 48 -> 16 kHz is a stateful FIR + decimate-by-3 (chunk-wise
# resample_poly rings at the edges). 2026-11-04.
_VAD_VIEWER_DIRECT = os.getenv("VAD_VIEWER_DIRECT", "0") == "1"
# 48 -> 16 kHz anti-alias low-pass for VAD_VIEWER_DIRECT (63 taps, 7 kHz).
_DEC_FIR = firwin(63, 7000, fs=48000)
_FAST_RELEASE_MIN_S = float(os.getenv("SPEC_FAST_RELEASE_MIN_S", "0.7") or 0.7)
# SMART_TURN (vad/smart_turn.py): release at the pause when the end-of-turn
# model is at least this sure the speaker has finished. Below it, nothing
# changes (the endpoint still decides): the model scored some complete test
# questions as incomplete (2026-11-04), so it is trusted only one way.
_SMART_TURN_RELEASE = float(os.getenv("SMART_TURN_RELEASE", "0.8") or 0.8)
# Addressee check (face_detect.ATTENTION): a turn is answered only if someone
# near the PROTO was facing it while it was spoken. Judged from camera frames
# already seen, so it adds nothing to the reply time; with no camera it never
# drops a turn. ADDRESSEE=0 turns it off (read in face_detect, which also tells
# the render tab, so barge-in follows the same switch).
# ADAPTIVE ENDPOINT tiers (paired with the shortened hard endpoint, ~0.45s):
# a transcript that clearly ENDS (。!? or a sentence-final form) fires right at
# the short endpoint; an AMBIGUOUS one waits this much longer (≈ the old 0.77s
# total) so mid-thought pauses aren't cut; incomplete ones hold _VAD_GRACE_S.
_AMBIG_GRACE_S = 0.30
# NOTE: the ASCII period IS in the class — without it every properly punctuated
# ENGLISH transcript ("Hello.") fell into the ambiguous tier and paid +0.3s.
_STRONG_END_RE = re.compile(
    r"([。．.!?！？♪]|です|ます|ました|ません|ですか|ますか|でしょう?|ください|"
    r"だよ|だね|よね|かな|なんだ|でした)$"
)
# Rapid-follow-up merge window: a VAD turn firing within this many seconds of
# the previous one (while its reply still runs) is folded into ONE combined
# turn instead of spawning a competing conversation.
_TURN_MERGE_S = 5.0
# Hold the barge-in while she speaks until the words are heard to be for her
# (_speech_onset_omni). BARGE_CONFIRM=0: stop her at the onset, as before.
_BARGE_CONFIRM = os.getenv("BARGE_CONFIRM", "1") == "1"
# Busy room (turn_contingency.busy_room): answer only what both transcripts heard.
_UNDERSTOOD_GATE = os.getenv("BUSY_UNDERSTOOD_GATE", "1") == "1"
_ASR_FAST_BUSY = os.getenv("ASR_FAST_BUSY", "1") == "1"
_UNDERSTOOD_MIN = float(os.getenv("BUSY_UNDERSTOOD_MIN", "0.5") or 0.5)
# How much of the speech the held barge-in is first judged on (_barge_probe).
_BARGE_PROBE_S = float(os.getenv("BARGE_PROBE_S", "0.7") or 0.7)
# Whisper's notorious Japanese hallucinations (YouTube-outro phrases emitted for
# noise/breath/silence segments). A phantom turn made of one of these fired the
# supersede logic and CUT the running reply's audio mid-answer ("字幕には出るの
# に調べた内容をしゃべらない"). Dropped only when the transcript is essentially
# just the phrase — embedded in long real speech it passes through.
_ASR_HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "チャンネル登録",
    "高評価",
    "次の動画でお会いしましょう",
    "字幕視聴ありがとうございました",
    # English whisper hallucinations (matched case-insensitively). The bare
    # "Thank you." is deliberately NOT here — users genuinely say it.
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "amara.org",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
)

# A transcript ending with sentence-final punctuation is complete (whisper
# reliably appends 。 to finished JA sentences). Otherwise, clear continuation
# markers — trailing comma, connective particles / conjunctions — mean the
# speaker is mid-thought. Notably "は" is EXCLUDED (こんにちは/こんばんは).
_COMPLETE_ENDINGS = ("。", "！", "？", "!", "?", ".", "…")
_CONTINUATION_ENDINGS = (
    "、", ",", "て", "で", "が", "けど", "けれど", "から", "ので", "のに",
    "し", "とか", "や", "に", "を", "も", "と", "たら", "れば", "って",
    # two-char topic endings that are safe from the こんにちは collision
    "のは", "には", "とは", "ては",
)
# English mid-thought endings need a WORD boundary (plain endswith("and") would
# match "band"). Trailing conjunctions/prepositions/articles/auxiliaries = the
# speaker is mid-sentence; a false positive only costs the 0.8s grace.
# (Only the fallback now: in omni mode the Thinker judges the grammar of any
# sentence — omni_addressee.COMPLETE, _unfinished below.)
_EN_CONTINUATION_RE = re.compile(
    r"\b(and|but|so|because|or|to|of|with|that|if|when|for|at|in|on|as|than|"
    r"the|a|an|my|your|our|their|his|her|its|is|are|was|were|will|would|"
    r"could|should|can|about|like|then|while|though|although|unless|until|"
    r"since)$",
    re.IGNORECASE,
)


# --- junk / noise rejection (2026-07-27) --------------------------------
# Room noise (a door, a chair, a cough, distant chatter) still reaches the ASR
# occasionally, and ASR ALWAYS returns *something* — usually a one-word
# fragment or a run of the same syllable. Answering those makes the avatar look
# like it talks to the room. A turn must carry real content to be answered.
# Tunable: VAD_MIN_CHARS (CJK) / VAD_MIN_WORDS (latin); 0 disables the check.
_JUNK_FILLERS = {
    # single interjections an ASR emits for a cough/breath/bump
    "ah", "aah", "ha", "hah", "huh", "hm", "hmm", "mm", "mmm", "uh", "um", "oh",
    "eh", "er", "yeah", "ya", "mhm", "uh-huh", "wow", "ow", "hey",
    "あ", "ああ", "あー", "え", "ええ", "えー", "ん", "んー", "う", "うー",
    "お", "おー", "は", "はー", "ふ", "ふー", "ねえ", "ねー",
    "うーん", "えーと", "えっと", "あのー", "あの", "そのー", "まあ", "まー",
    "へえ", "へー", "ほう", "ふーん", "はあ", "ふう", "おお", "うわ", "あれ",
}
_CJK_CHARS_RE = re.compile(r"[぀-ヿ一-鿿]")
# Words that ARE a legitimate whole turn on their own. Takes precedence over
# _JUNK_FILLERS (a lone "Yeah." is an answer, not a cough) and over the
# minimum-word-count rule. The noise-words the ASR invents from room sounds
# ("And.", "Out.", "Ii.") are deliberately absent.
_REAL_SHORT_WORDS = {
    "yes", "yeah", "yep", "yup", "no", "nope", "nah",
    "stop", "wait", "hello", "hi", "hey", "bye", "goodbye",
    "thanks", "thank", "sorry", "ok", "okay", "sure", "please",
    "continue", "repeat", "again", "right", "correct", "exactly",
    "maybe", "understood", "affirmative", "negative", "cool",
    "nice", "good", "great", "perfect", "what", "why", "how",
    "who", "when", "where", "really",
}


# Short courtesy phrases faster-whisper falls back on when a segment carries no
# speech. Each one is something a user genuinely SAYS, so a single occurrence is
# always kept — only repetition across a turn is rejected (see _is_looped_noise).
_ASR_LOOP_PHRASES = {
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thanks a lot", "bye", "goodbye", "okay", "ok", "you", "so", "oh",
    "ありがとう", "ありがとうございます", "ありがとうございました",
    "はい", "おはよう", "こんにちは",
}


# CUMULATIVE ASR (2026-10-23). Every segment pause used to send whisper only the
# new segment — often under a second of audio, cut out by the VAD, with no words
# around it — and the turn was those transcripts glued together. Captured at the
# PROTO, the same speech came out right when transcribed whole and wrong in
# pieces: "Can you search about UNO AI Corp?" -> "All of you, may I go home?",
# "Hi, how are you?" -> "Mike, how are you?", and noise-only pieces added "Thank
# you." / "Bye." / "Thank you for watching." to real sentences. That is not one
# word but every word: whisper decodes from context, and the pieces had none.
# Now each pause re-transcribes the WHOLE utterance so far from the raw mic
# stream (pauses and word onsets included), and the turn is the newest such
# transcript. The last pause comes ~0.7 s before the endpoint, so an utterance
# of a few seconds is still transcribed by the time the turn ends.
# CUMULATIVE_ASR=0 restores per-segment transcription.
_CUMULATIVE_ASR = os.getenv("CUMULATIVE_ASR", "1") == "1"
_ASR_MAX_UTTERANCE = 28 * 16000  # whisper's window is 30 s
_MIC_RING_SAMPLES = 60 * 16000


class _MicRing:
    """A client's raw 16 kHz mic stream, the last minute of it, by sample index
    (and when each chunk arrived, to find the same moment in another stream)."""

    def __init__(self) -> None:
        self.chunks: deque[tuple[int, np.ndarray, float]] = deque()
        self.end = 0  # sample index just past the newest chunk

    def append(self, frame: np.ndarray) -> None:
        self.chunks.append((self.end, frame, time.time()))
        self.end += len(frame)
        while self.chunks and self.end - self.chunks[0][0] > _MIC_RING_SAMPLES:
            self.chunks.popleft()

    def slice(self, start: int, end: int) -> np.ndarray:
        parts = [
            a[max(0, start - s) : max(0, end - s)]
            for s, a, _ in self.chunks
            if s < end and s + len(a) > start
        ]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def wall(self, index: int) -> float:
        """Arrival time (time.time()) of the sample at `index`."""
        for s, a, t in self.chunks:
            if s <= index < s + len(a):
                return t - (s + len(a) - index) / 16000
        return self.chunks[0][2] if self.chunks else time.time()


# The render tab's copy of the PROTO mic lags the original by ~0.1 s (81-111 ms
# by cross-correlation); start the original that much earlier, plus a margin.
_VIEWER_LEAD_S = 0.15

# omni mode (KEHAI_MODE=omni, from .mode via start_server.sh): the turn's audio
# itself is handed to the speech-in LLM (Qwen3-Omni) next to Whisper's text.
# Whisper keeps the turn control (segments, semantic endpoint, speculation,
# history); only the understanding moves to the audio.
_OMNI_AUDIO_IN = os.getenv("KEHAI_MODE", "cascade") == "omni"


def _wav_b64_16k(audio: np.ndarray) -> str:
    """float32 16 kHz mono -> base64 of a 16-bit WAV (what an `audio_url`
    data: part carries)."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]*\s*")


def _turn_parts(results: list) -> list[str]:
    """The pieces a turn's text is built from, given its ASR tasks' results.

    Per-segment ASR: each segment's transcript. Cumulative ASR: the newest
    finished transcript already covers the whole utterance, so its SENTENCES —
    still one unit each for the filler checks below, which were written for
    segments: a noise tail whisper turns into "Thank you for watching." is
    dropped as a sentence without taking the real sentence before it along.
    Pieces keep their trailing whitespace, so "".join() rebuilds the text.
    """
    texts = [r for r in results if isinstance(r, str) and r.strip()]
    if not _CUMULATIVE_ASR:
        return [t.strip() for t in texts]
    if not texts:
        return []
    if results and isinstance(results[-1], Exception):
        # The newest transcript (the whole utterance) failed even on retry; the
        # last one that worked ends at an earlier pause.
        logger.warning(
            f"[server-vad] newest ASR failed ({results[-1]!r}); using the previous "
            f"pause's transcript: {texts[-1][-60:]!r}"
        )
    return [p for p in _SENTENCE_RE.findall(texts[-1].strip()) if p.strip()]


def _segment_is_hallucination(segment: str) -> bool:
    """True when ONE ASR segment is whisper's non-speech filler.

    Applied per segment, before the turn's segments are joined. The turn-level
    test below is gated on `len(text) <= 40`, which worked when a turn was one
    segment — but a turn is the CONCATENATION of every segment in it, and three
    hallucinated segments in a row ("Thank you." + "Thanks." + "Thank you very
    much." = 36 chars, or the same with "Thank you for watching." = 52) either
    sail past the length gate or read as one long sentence. Each segment is its
    own whisper call, so the segment is the unit the gate was written to judge.
    """
    s = segment.strip().lower()
    return bool(s) and any(h in s for h in _ASR_HALLUCINATIONS)


def _is_looped_noise(segments: list[str]) -> bool:
    """True when a turn is whisper idling on room tone rather than speech.

    faster-whisper answers silence with a small fixed vocabulary, and over a run
    of non-speech segments it emits phrases from it every time. A person says
    "Thank you." once; three segments of nothing but courtesy phrases in a single
    turn is the decoder spinning. Requires >= 3 segments and every one of them to
    be a bare phrase, so "Thank you. That was helpful." (2 segments, the second
    with real content) and a lone "Thanks." are both untouched.

    Written from the 2026-08-13 incident: a video playing in the room produced
    736 turns in a day, each one a full LLM call, and the replies to them were
    what started the "..." spiral.
    """
    norm = [re.sub(r"[\s.,!?！？。、…]+", " ", s).strip().lower() for s in segments]
    norm = [n for n in norm if n]
    return len(norm) >= 3 and all(n in _ASR_LOOP_PHRASES for n in norm)


def _is_junk_transcript(text: str) -> bool:
    """True when the turn is almost certainly noise rather than speech aimed at
    the avatar. Conservative: anything with real sentence content passes."""
    t = re.sub(r"[\s.,!?！？。、…・~〜\-]+", " ", text).strip()
    if not t:
        return True
    core = t.lower().replace(" ", "")
    # a lone interjection ("Ah.", "うーん") — or the same one repeated ("ah ah ah")
    words = [w for w in t.lower().split() if w]
    if words and any(w.strip("-") in _REAL_SHORT_WORDS for w in words):
        return False  # a real word is present — this is speech, not a cough
    if words and all(w.strip("-") in _JUNK_FILLERS for w in words):
        return True
    if core in _JUNK_FILLERS:
        return True
    # single repeated character run ("あああ", "aaaa")
    if len(set(core)) == 1 and len(core) >= 2:
        return True
    is_cjk = bool(_CJK_CHARS_RE.search(t))
    if is_cjk:
        min_chars = int(os.getenv("VAD_MIN_CHARS", "2"))
        return min_chars > 0 and len(_CJK_CHARS_RE.findall(t)) < min_chars
    min_words = int(os.getenv("VAD_MIN_WORDS", "2"))
    if min_words <= 0:
        return False
    # 1-word turns are junk UNLESS they are a real reply/command. Observed
    # noise-words the ASR invents ("And.", "Out.", "Ii.") are NOT in this list,
    # while genuine one-word answers the user does say ("Yeah.", "Bye.") are —
    # dropping those was making real turns vanish.
    if len(words) < min_words:
        return words[0] not in _REAL_SHORT_WORDS
    return False


def _transcript_looks_incomplete(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return False
    # Whisper writes a trailing ELLIPSIS when the speaker trails off mid-thought
    # ("...because I want to..."). That is the opposite of a finished sentence,
    # but it ends in "." so the complete-endings test below would wave it
    # through and fire the turn while you were still talking.
    if t.endswith("...") or t.endswith("…"):
        return True
    if t.endswith(_COMPLETE_ENDINGS):
        return False
    if t.endswith(_CONTINUATION_ENDINGS):
        return True
    return bool(_EN_CONTINUATION_RE.search(t))


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        # Server-VAD continuous-mic path: per-client in-flight segment ASR tasks
        # (each VAD-pause-delimited segment is transcribed while the user is
        # still talking; the turn end only waits for the last short one).
        self.segment_asr_tasks: Dict[str, list] = {}
        # Cumulative ASR: each client's recent mic stream, and where the current
        # utterance starts in it (sample index, set by its first segment).
        self.mic_rings: Dict[str, _MicRing] = {}
        # omni: (ring index, monotonic) where the previous turn's utterance
        # began — a rapid follow-up merge re-cuts its audio from there, so the
        # Thinker hears both halves (it heard only the second one before,
        # 2026-11-04 review).
        self.last_turn_start: Dict[str, tuple[int, float]] = {}
        self.prev_turn_start: Dict[str, tuple[int, float] | None] = {}
        # VAD_VIEWER_DIRECT: which render client owns render_hub.mic_sink, and
        # its 48->16 kHz decimator state.
        self._mic_sink_uid: str | None = None
        self._dec_state: Dict[str, dict] = {}
        self.utterance_start: Dict[str, int] = {}
        # omni mode: the finished utterance (base64 WAV) waiting for its turn to
        # be finalized — taken at <|TURN_END|>, consumed by _finalize_vad_turn.
        self.turn_audio: Dict[str, str] = {}
        # Clients that are the render tab (/client-ws?render=1): their mic is the
        # PROTO's, whose original the server also has (render_stream.hub).
        self.render_clients: set[str] = set()
        # Semantic endpointing: a turn whose transcript ends mid-thought is held
        # here for a grace period; resuming speech folds it into the next turn.
        self.pending_vad_transcript: Dict[str, str] = {}
        self.vad_grace_tasks: Dict[str, asyncio.Task] = {}
        # Rapid-follow-up merge: (fired_at_monotonic, text) of the last VAD turn.
        # A new turn that lands while the previous reply is still running AND
        # within _TURN_MERGE_S is treated as the SAME utterance split by a pause
        # (完結に見える途中文で VAD が切った) — the texts merge and one combined
        # reply supersedes the stale one.
        self.last_vad_turn: Dict[str, tuple] = {}
        # When the client last interrupted her (barge-in). A turn that follows
        # within _TURN_MERGE_S is treated as the CONTINUATION of the one she was
        # answering: with the endpoint at ~0.47 s she starts replying inside
        # people's own pauses, they talk on, the barge-in cuts her, and the
        # fragment ("right now in Omaha?") was answered alone (2026-11-04 08:21).
        self.interrupted_at: Dict[str, float] = {}
        # When her last reply ended (monotonic) and what it said: a short
        # answer right after it ("Yes.", "Why?") is a reply to her even when it
        # sounds like a fragment on its own (see _reply_context).
        self.last_reply: Dict[str, tuple] = {}
        # A barge-in held at the onset because she was speaking (monotonic onset
        # time): sent once the words are heard to be for her, dropped otherwise
        # (_speech_onset_omni holds it; _confirm_barge / _drop_barge settle it).
        self.barge_held: Dict[str, float] = {}
        self._asr_locks: Dict[int, asyncio.Lock] = {}  # per session context (_asr_one_at_a_time)
        # Between a speech onset and its <|TURN_END|>: someone is talking now.
        self.in_utterance: Dict[str, bool] = {}
        # When the last <|TURN_END|> fired, and the <|TURN_END|> of the last turn
        # she took (answered or held for more) — see _CONT_S.
        self.turn_end_at: Dict[str, float] = {}
        self.accepted_at: Dict[str, float] = {}
        self.session_open: Dict[str, bool] = {}
        self._bg_checked: Dict[str, int] = {}
        self._visitor_fresh: set = set()
        turn_contingency.SHE = lambda: bool(render_hub.speaking)
        turn_contingency.ALONE = lambda: face_detect.ATTENTION.empty_for(3.0)
        voice_direction.SHE = lambda: bool(render_hub.speaking)
        self.invite_at: Dict[str, float] = {}
        self._invite_clip: Dict[str, str] = {}
        self._last_nod: Dict[str, float] = {}
        # epoch of a turn whose reply smart-turn released before the endpoint
        self.early_released: Dict[str, int] = {}
        # Speculative reply per client, and the lock that serialises starting,
        # confirming and discarding one: a discard rolls the history back to its
        # checkpoint, so it must never interleave with a real turn's first writes.
        self.speculations: Dict[str, speculative.Speculation] = {}
        self.spec_locks: Dict[str, asyncio.Lock] = {}
        # Bumped whenever the user starts or resumes speaking. A speculation
        # remembers the value at its segment close; if it moved while the ASR
        # ran, the user is already talking again and the reply would be stale.
        self.speech_epoch: Dict[str, int] = {}
        # Monotonic time the current utterance began (IDLE->ACTIVE), for the
        # addressee check. A held mid-thought turn keeps its original start.
        self.turn_started: Dict[str, float] = {}
        # MIC LIVENESS. The WebRTC mic track can stop delivering media while the peer
        # still reports connectionState 'connected' — every signal the BROWSER can see
        # still looks healthy, so its own disconnected/failed self-heal never fires and
        # the avatar simply stops answering ("途中から話しかけても喋らなくなる").
        # Only this side can tell: mic frames arrive continuously while the track is
        # live, so a gap means the stream is dead, not that the room went quiet.
        self.last_mic_chunk: Dict[str, float] = {}
        self.mic_watchdogs: Dict[str, asyncio.Task] = {}
        # Rebuild requests already sent for the current stall. Kept OUTSIDE the watchdog
        # task because each rebuild starts a fresh one, so a counter local to the task
        # would reset every attempt and hammer a genuinely absent microphone forever.
        self.mic_stall_asks: Dict[str, int] = {}

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "webrtc-offer": self._handle_webrtc_offer,
            "motion-usage": self._handle_motion_usage,
            "audio-health": self._handle_audio_health,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "audio-caps": self._handle_audio_caps,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        from .utils import loop_lag  # LOOP_LAG=1: event-loop lag in the log

        loop_lag.start()
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except WebSocketDisconnect:
            # The ordinary case: a reload closes the socket while the handshake
            # is still running. Logged as an ERROR with an EMPTY message before
            # (str(WebSocketDisconnect) is ""), which read like a real fault with
            # no detail — two of them on every page refresh.
            logger.info(f"Client {client_uid} closed the socket during setup")
            await self._cleanup_failed_connection(client_uid)
            raise
        except Exception as e:
            # !r, not str: several exceptions here stringify to nothing at all,
            # so the type is the only thing that identifies them.
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e!r}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # CONTINUITY: resume this character's most recent conversation instead
        # of starting blank. Without this every reconnect (or app restart) wiped
        # the thread — the history file was written but never read back, so she
        # could not refer to anything you had said before.
        self._resume_latest_history(session_service_context)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=(
                self.default_context_cache.vad_engine.fresh()
                if hasattr(self.default_context_cache.vad_engine, "fresh")
                else self.default_context_cache.vad_engine
            ),
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        await webrtc_manager.close(client_uid)
        _ws = self.client_connections.get(client_uid)
        if _ws is not None:
            ws_opus.forget_client(_ws)
        wd = self.mic_watchdogs.pop(client_uid, None)
        if wd is not None and not wd.done():
            wd.cancel()
        self.last_mic_chunk.pop(client_uid, None)
        self.mic_stall_asks.pop(client_uid, None)
        self.mic_rings.pop(client_uid, None)
        self.utterance_start.pop(client_uid, None)
        self.turn_audio.pop(client_uid, None)
        self.render_clients.discard(client_uid)
        if self._mic_sink_uid == client_uid:
            render_hub.mic_sink = None
            render_hub.text_sink = None
            self._mic_sink_uid = None
        self._dec_state.pop(client_uid, None)
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        context = self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient). The
        # context is the one popped above: the old `.get()` after that pop
        # always found None, so a session's MCP server processes were never
        # closed on disconnect.
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        context = self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # A context that got as far as _init_mcp_components may already own
        # pre-spawned MCP server processes.
        if context:
            try:
                await context.close()
            except Exception as e:  # noqa: BLE001 — cleanup of a failed connection
                logger.warning(f"Closing context of failed connection: {e!r}")

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        self.interrupted_at[client_uid] = time.monotonic()
        if client_uid in self.render_clients and render_hub.direct_pcm:
            await render_hub.viewer_flush("interrupt-signal")
        # Barge-in: drop any AI speech still queued on the WebRTC track so the
        # voice cuts off immediately (WS streaming stops via the task cancel).
        webrtc_manager.clear_audio(client_uid)
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Continuous-mic path (server-side VAD).

        The client streams every 32ms mic frame; Silero VAD segments the stream
        at pauses. Each pause-delimited SEGMENT is transcribed (gateway ASR)
        while the user is still talking, so the end of the turn only waits for
        the LAST short segment instead of a whole-utterance batch ASR. On
        <|TURN_END|> the accumulated transcript triggers the conversation
        directly through the text-input path (no batch ASR step at all).
        """
        chunk = data.get("audio", [])
        if not chunk:
            return
        if _VAD_VIEWER_DIRECT and client_uid in self.render_clients:
            # The viewer's own frames feed the VAD (render_hub.mic_sink); the
            # tab's copy of them only marks the mic as alive.
            if render_hub.mic_sink is None or self._mic_sink_uid != client_uid:
                render_hub.mic_sink = functools.partial(self._viewer_mic_pcm, websocket, client_uid)
                render_hub.text_sink = functools.partial(self._typed_to_render, websocket, client_uid)
                self._mic_sink_uid = client_uid
                self._dec_state.pop(client_uid, None)
                logger.info(f"[vad] render client {client_uid[:8]}: server VAD now runs on the viewer's mic directly")
            self.last_mic_chunk[client_uid] = time.monotonic()
            return
        await self._process_vad_chunk(websocket, client_uid, chunk)

    async def _typed_to_render(self, websocket: WebSocket, client_uid: str, text: str) -> None:
        """/render-stream/say: typed text, taken as the visitor's turn at the PROTO
        (no addressee check — whoever typed it meant her)."""
        if client_uid not in self.client_contexts:
            return
        await self._handle_conversation_trigger(websocket, client_uid, {"type": "text-input", "text": text})

    async def _viewer_mic_pcm(self, websocket: WebSocket, client_uid: str, pcm: bytes) -> None:
        """VAD_VIEWER_DIRECT: a 48 kHz int16 chunk from the viewer -> 16 kHz
        float32 -> 512-sample frames into the server VAD."""
        if client_uid not in self.client_contexts:
            return
        st = self._dec_state.get(client_uid)
        if st is None:
            st = self._dec_state[client_uid] = {
                # zero initial state (lfilter_zi solved a 62x62 system just to be
                # multiplied by 0, and its first call blocked the loop 401 ms)
                "zi": np.zeros(len(_DEC_FIR) - 1, dtype=np.float64),
                "phase": 0,
                "buf": np.zeros(0, dtype=np.float32),
            }
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return
        y, st["zi"] = lfilter(_DEC_FIR, [1.0], x, zi=st["zi"])
        dec = y[st["phase"]::3]
        st["phase"] = (st["phase"] + 3 * len(dec)) - len(y)
        buf = np.concatenate([st["buf"], dec.astype(np.float32)])
        n = 512
        while len(buf) >= n:
            frame, buf = buf[:n], buf[n:]
            await self._process_vad_chunk(websocket, client_uid, frame)
        st["buf"] = buf

    async def _handle_audio_health(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Log what the browser did with the turn's audio.

        The server can prove it SENT the voice; nothing here could show whether
        the page decoded, scheduled or played it, so "she has no voice" always
        came down to asking someone to open a console mid-conversation. `ctx`
        settles it: 'running' puts the fault past this app (output device, muted
        tab, OS mixer), 'suspended' means the browser is holding the audio until
        the page gets a user gesture — and speaking into the mic is not one.
        Pure measurement; nothing branches on it."""
        ctx = str(data.get("ctx", "?"))
        note = ""
        if ctx == "suspended":
            note = "  <-- BROWSER IS BLOCKING PLAYBACK: the page needs one click"
        elif ctx == "running" and float(data.get("seconds", 0) or 0) > 0:
            note = "  (samples reached the output — look at the device/tab/OS mixer)"
        logger.info(
            f"[audio-health] {client_uid[:8]} ctx={ctx} "
            f"buffers={data.get('buffers', '?')} secs={data.get('seconds', '?')} "
            f"rate={data.get('rate', '?')} gain={data.get('gain', '?')} "
            f"preroll={data.get('preroll', '?')} gaps={data.get('gaps', '?')}"
            f"/{data.get('gapMs', '?')}ms@{data.get('gapAt', '?')}s origin={data.get('origin', '?')}{note}"
        )

    async def _handle_motion_usage(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Log the frontend's per-turn motion ledger: how many seconds each layer
        (generated clip / clip-hold / procedural talking / idle) actually drove
        the avatar. Pure measurement — no behavior."""
        clip = float(data.get("clip", 0) or 0)
        hold = float(data.get("hold", 0) or 0)
        proc = float(data.get("proc", 0) or 0)
        idle = float(data.get("idle", 0) or 0)
        applies = int(data.get("applies", 0) or 0)
        total = clip + hold + proc + idle
        if total <= 0.5:
            return

        def pct(x: float) -> str:
            return f"{x / total * 100:.0f}%"

        logger.info(
            f"[motion-usage] clip {clip:.1f}s({pct(clip)}) | "
            f"hold {hold:.1f}s({pct(hold)}) | procedural {proc:.1f}s({pct(proc)}) | "
            f"idle {idle:.1f}s({pct(idle)}) | applies {applies} | "
            f"gesture-frames {data.get('gframes', '?')} | "
            # `lsm` is the one that answers "is co-speech motion actually driving
            # the body": GestureLSM upgrades that LANDED, not ones scheduled.
            f"lsm-applied {data.get('lsm', '?')} | vrm {data.get('vrmOk', '?')}"
        )

    async def _handle_webrtc_offer(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Phase G signaling: answer the browser's WebRTC offer. Its mic track
        feeds the SAME server-VAD segment pipeline as raw-audio-data; the
        answer's outbound track carries the streamed TTS PCM (no preroll, no
        base64, and the browser AEC gets a true render reference)."""
        sdp = data.get("sdp")
        if not sdp:
            return

        async def on_mic_chunk(chunk) -> None:
            # (liveness is stamped inside _process_vad_chunk, for both transports)
            if _VAD_VIEWER_DIRECT and client_uid in self.render_clients and render_hub.mic_sink is not None:
                # The viewer's own frames already run this client's VAD: the
                # tab's copy over WebRTC would feed every sample a second time,
                # ~0.1 s late, into the same detector (raw-audio-data has the
                # same guard). Not seen live — the render tab has no WebRTC
                # peer today — but nothing prevented it.
                self.last_mic_chunk[client_uid] = time.monotonic()
                return
            await self._process_vad_chunk(websocket, client_uid, chunk)

        self._start_mic_watchdog(websocket, client_uid)

        try:
            answer = await webrtc_manager.handle_offer(
                websocket, client_uid, sdp, on_mic_chunk
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[webrtc] offer handling failed: {e!r}")
            return
        await websocket.send_text(
            json.dumps({"type": "webrtc-answer", "sdp": answer})
        )

    # No mic frame for this long => the track is dead, not the room quiet. Frames arrive
    # every ~20ms while the track lives, so this is far above any legitimate gap; it is
    # kept generous so a GC pause or a tab throttle cannot trigger a needless rebuild.
    _MIC_STALL_S = 6.0

    def _start_mic_watchdog(self, websocket: WebSocket, client_uid: str) -> None:
        """Ask the client to rebuild its peer if its mic stops delivering audio.

        Called on EVERY offer, including the ones this watchdog itself asked for.
        That is why it must not restart itself: it used to cancel the running
        task and stamp `last_mic_chunk = now`, which meant a rebuild forged proof
        of liveness. The 2s tick then saw `gap < 6.0`, popped `mic_stall_asks`
        back to zero, and the next stall was "attempt 1/3" again — for ever. The
        `warned_at` throttle was a task-local, so re-creating the task reset that
        too. Net effect: the "3 rebuilds then stop" rule never once fired, and a
        client that could not complete ICE (every client behind an SSH port
        forward) was told to rebuild its peer every ~13 seconds indefinitely.

        So: one watchdog per client, armed once, surviving every renegotiation.
        """
        old = self.mic_watchdogs.get(client_uid)
        if old is not None and not old.done():
            return  # already watching this client — keep its counters
        # A baseline only if there is none. NOT a refresh: the stamp means "mic
        # audio last arrived at", and answering an offer is not mic audio.
        self.last_mic_chunk.setdefault(client_uid, time.monotonic())

        async def _watch() -> None:
            warned_at = 0.0
            try:
                while True:
                    await asyncio.sleep(2.0)
                    if client_uid not in self.client_connections:
                        return
                    last = self.last_mic_chunk.get(client_uid)
                    if last is None:
                        continue
                    gap = time.monotonic() - last
                    if gap < self._MIC_STALL_S:
                        # Audio is flowing again — this stall is over, so a LATER one
                        # gets a full set of attempts rather than inheriting these.
                        self.mic_stall_asks.pop(client_uid, None)
                        continue
                    # Re-ask periodically rather than once: a rebuild can itself fail,
                    # and one unheeded nudge would leave the mic dead for the session.
                    if time.monotonic() - warned_at < 15.0:
                        continue
                    asks = self.mic_stall_asks.get(client_uid, 0)
                    if asks >= 3:
                        # Three rebuilds did not bring audio back, so the microphone is
                        # gone (unplugged, taken by another app, permission revoked)
                        # rather than the stream being stuck. Rebuilding on a loop would
                        # just churn getUserMedia; leave it and say so once.
                        if asks == 3:
                            self.mic_stall_asks[client_uid] = 4
                            logger.warning(
                                f"[mic-watchdog] {client_uid}: 3 rebuilds did not "
                                "restore audio — the microphone itself looks gone"
                            )
                        continue
                    self.mic_stall_asks[client_uid] = asks + 1
                    warned_at = time.monotonic()
                    logger.warning(
                        f"[mic-watchdog] no audio from {client_uid} for {gap:.1f}s "
                        f"— asking the client to rebuild its WebRTC peer "
                        f"(attempt {asks + 1}/3)"
                    )
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "control", "text": "restart-mic"})
                        )
                    except Exception as e:  # noqa: BLE001 — client already gone
                        logger.info(f"[mic-watchdog] could not reach client: {e}")
                        return
            except asyncio.CancelledError:
                return

        self.mic_watchdogs[client_uid] = asyncio.create_task(_watch())

    async def _process_vad_chunk(
        self, websocket: WebSocket, client_uid: str, chunk
    ) -> None:
        """Shared server-VAD core for both upstream transports (WS float frames
        and the WebRTC mic track)."""
        # Mic liveness is recorded HERE, at the one point both transports pass
        # through. It used to be stamped only in the WebRTC mic callback, so a
        # client that had fallen back to WS audio looked mute to the watchdog no
        # matter how well its microphone worked: three pointless "rebuild your
        # peer" requests followed by "the microphone itself looks gone", every
        # session, while the transcripts kept arriving. A watchdog that lies
        # about the thing it watches is worse than no watchdog when something
        # else is actually broken.
        self.last_mic_chunk[client_uid] = time.monotonic()
        context = self.client_contexts.get(client_uid)
        if context is None or context.vad_engine is None:
            return
        frame = np.asarray(chunk, dtype=np.float32)
        ring = self.mic_rings.setdefault(client_uid, _MicRing())
        ring.append(frame)
        if asr_capture.on():
            asr_capture.record(
                "tab", (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
            )
        for audio_bytes in context.vad_engine.detect_speech(chunk):
            if audio_bytes == b"<|PAUSE|>":
                # New turn began (IDLE->ACTIVE). Drop any tasks left over from
                # an aborted turn (e.g. mic toggled off mid-utterance) so they
                # can't leak into this turn's transcript. Barge-in interruption
                # stays client-driven (its own VAD already interrupts).
                self.segment_asr_tasks.pop(client_uid, None)
                self.utterance_start.pop(client_uid, None)
                now = time.monotonic()
                self.in_utterance[client_uid] = True
                if active_speaker.ENABLED and _OMNI_AUDIO_IN and client_uid in self.render_clients:
                    _spawn(self._asd_prepare(client_uid))
                if client_uid not in self.pending_vad_transcript:
                    self.turn_started[client_uid] = now
                    await self._maybe_new_visitor(websocket, client_uid)
                # Visitors talking among themselves while she speaks must not
                # strip the gestures off the rest of her reply. (Unjudged —
                # no camera — cancels as before.)
                aside = face_detect.ADDRESSEE and face_detect.ATTENTION.judge(now, now)[0] is False
                # omni mode: the Thinker looks at the camera frame now (~0.1 s,
                # omni_addressee). The barge-in signal below waits for that look
                # — bounded — so a visitor talking to someone else does not stop
                # her; with nothing to look at (no camera) it goes out as before.
                if self._omni_judged(client_uid) and omni_addressee.JUDGE.check("onset") is not None:
                    _spawn(self._speech_onset_omni(websocket, client_uid, now, aside))
                else:
                    await self._speech_onset_actions(websocket, aside)
                # Semantic endpointing: the user resumed while an incomplete
                # transcript was in its grace window — cancel the pending fire;
                # the held text will prepend to this new turn's transcript.
                grace = self.vad_grace_tasks.pop(client_uid, None)
                if grace is not None and not grace.done():
                    grace.cancel()
                self.speech_epoch[client_uid] = self.speech_epoch.get(client_uid, 0) + 1
                _spawn(self._drop_speculation(client_uid, "new turn"))
            elif audio_bytes == b"<|SPEECH|>":
                self.speech_epoch[client_uid] = self.speech_epoch.get(client_uid, 0) + 1
                _spawn(self._drop_speculation(client_uid, "user resumed"))
            elif audio_bytes == b"<|RESUME|>":
                pass
            elif audio_bytes.startswith(b"<|SEG|>"):
                if context.asr_engine is None:
                    continue
                pcm = np.frombuffer(audio_bytes[len(b"<|SEG|>") :], dtype=np.int16)
                segment = pcm.astype(np.float32) / 32768.0
                self._maybe_listening_nod(websocket, client_uid)
                logger.debug(
                    f"[server-vad] segment closed ({len(segment) / 16000:.2f}s) "
                    "-> ASR while user keeps talking"
                )
                audio, viewer_span = segment, None
                if _CUMULATIVE_ASR:
                    # The whole utterance so far, straight from the mic stream —
                    # pauses and word onsets included (see _CUMULATIVE_ASR).
                    start = max(
                        self.utterance_start.setdefault(client_uid, ring.end - len(pcm)),
                        ring.end - _ASR_MAX_UTTERANCE,
                    )
                    audio = ring.slice(start, ring.end)
                    if client_uid in self.render_clients and render_hub.viewer_mic_live():
                        viewer_span = (ring.wall(start) - _VIEWER_LEAD_S, time.time())
                    # Only the newest transcript is used; free the GPU of older ones.
                    for old in self.segment_asr_tasks.get(client_uid, []):
                        if not old.done():
                            old.cancel()
                # Busy room: the transcript is what tells babble from words
                # (_understood_in_busy_room) and must be in BEFORE the endpoint,
                # or she starts a reply and takes it back (18 false starts in 15
                # min, 2026-11-13). The quick model makes it; large-v3/beam 5 never did.
                _fast = _ASR_FAST_BUSY and client_uid in self.render_clients and turn_contingency.busy_room()
                asr_task = asyncio.create_task(self._transcribe(context, audio, viewer_span, fast=_fast))
                self.segment_asr_tasks.setdefault(client_uid, []).append(asr_task)
                if not omni_addressee.AUDIO_ENABLED and self._omni_judged(client_uid) and omni_addressee.JUDGE.judge(
                    self.turn_started.get(client_uid, time.monotonic()), time.monotonic()
                )[0] is not True:
                    # Another look only while this utterance is not yet judged "to
                    # her": a look fired on the endpoint's own SEG-END could make
                    # settle() wait up to 0.3 s, and each one shares GPU1 with the
                    # speculative prefill.
                    omni_addressee.JUDGE.check("segment")
                if _SPECULATE:
                    _spawn(
                        self._speculate(
                            websocket, client_uid, list(self.segment_asr_tasks[client_uid]),
                            self.speech_epoch.get(client_uid, 0),
                        )
                    )
                    if smart_turn.ENABLED and _FAST_RELEASE and _OMNI_AUDIO_IN:
                        _spawn(
                            self._smart_turn_check(websocket, client_uid, self.speech_epoch.get(client_uid, 0))
                        )
            elif audio_bytes == b"<|TURN_END|>":
                self.in_utterance[client_uid] = False
                self.turn_end_at[client_uid] = time.monotonic()
                # omni mode: keep the utterance audio for the turn about to be
                # finalized — the ring keeps rolling and utterance_start is
                # cleared right below.
                if _OMNI_AUDIO_IN:
                    _turn_wav = self._utterance_audio_b64(client_uid)
                    if _turn_wav:
                        self.turn_audio[client_uid] = _turn_wav
                    # For a rapid follow-up merge: where this utterance began in
                    # the mic ring, so the merged turn's audio can start here.
                    _us = self.utterance_start.get(client_uid)
                    if _us is not None:
                        # prev = the turn BEFORE this one: the merge for this
                        # turn must start the audio there, not at this
                        # utterance's own start (first test re-cut only the
                        # second half, 2026-11-04 08:24).
                        self.prev_turn_start[client_uid] = self.last_turn_start.get(client_uid)
                        self.last_turn_start[client_uid] = (_us, time.monotonic())
                self.utterance_start.pop(client_uid, None)
                tasks = self.segment_asr_tasks.pop(client_uid, [])
                # No segment this turn, but a held mid-thought one is waiting: the
                # sound that cancelled its grace was too short or too quiet to
                # become a segment. Finalize anyway so the held text is answered
                # now — skipped, it sat in pending_vad_transcript with no reply
                # and no chain-end, and was prepended to whatever came next.
                if tasks or self.pending_vad_transcript.get(client_uid):
                    _spawn(
                        self._finalize_and_settle(
                            websocket, client_uid, tasks,
                            self.speech_epoch.get(client_uid, 0),
                        )
                    )

    async def _release_spec(self, websocket: WebSocket, client_uid: str, spec, why: str, utter_s: float) -> bool:
        """Publish a speculative reply now, before its transcript (fast release
        at the endpoint, or earlier on a smart-turn "complete"). The transcript
        is filled in later by _speculate_inner, which also retracts the reply if
        the words turn out to be noise. True if this call released it."""
        async with self._spec_lock(client_uid):
            if self.speculations.get(client_uid) is not spec or spec.task.done():
                return False
            self.speculations.pop(client_uid, None)
            spec.released = True
            if not spec.gate.has_audio:
                await self._send_backchannel(websocket, client_uid)
            await spec.gate.release()
        self._watch_reply(client_uid, spec.task)
        self._mark_accepted(client_uid)
        head_ms = (time.monotonic() - spec.started) * 1000
        logger.info(
            f"[spec] FAST HIT: released {why}, reply running {head_ms:.0f} ms, "
            f"{spec.gate.held} msgs released, transcript "
            f"{'in' if spec.text_ready.is_set() else 'still pending'} ({utter_s:.1f}s utterance)"
        )
        self.last_vad_turn[client_uid] = (time.monotonic(), spec.text if spec.text_ready.is_set() else "")
        if spec.text_ready.is_set() and spec.text and not spec.placeholder:
            try:
                await websocket.send_text(json.dumps({"type": "user-input-transcription", "text": spec.text}))
            except Exception:  # noqa: BLE001
                pass
        return True

    async def _smart_turn_check(self, websocket: WebSocket, client_uid: str, epoch: int) -> None:
        """At a pause: ask the end-of-turn model (vad/smart_turn.py) whether the
        speaker has finished. Confident "complete" (>= SMART_TURN_RELEASE) and
        nothing has changed since (no new speech, no merge pending, addressed to
        her): release the speculative reply now — its first audio goes out the
        moment it exists instead of at the VAD endpoint. Anything else changes
        nothing; the endpoint stays the fallback. Every prediction is logged
        ([smart-turn]) so the threshold can be tuned on real PROTO speech."""
        try:
            # Let the speculative request leave first: this check must never
            # delay it (preparing the audio here on the loop pushed the request
            # back by ~20 ms). The audio is cut and resampled off the loop.
            await asyncio.sleep(0.03)
            if self.speech_epoch.get(client_uid, 0) != epoch:
                return
            start = self.utterance_start.get(client_uid)
            audio = await asyncio.to_thread(self._audio_np_from, client_uid, start)
            if audio is None:
                return
            utter_s = len(audio) / 16000
            p, ms = await smart_turn.predict_async(audio)
            if p is None:
                return
            act = p >= _SMART_TURN_RELEASE and utter_s >= _FAST_RELEASE_MIN_S
            logger.info(f"[smart-turn] P(complete)={p:.2f} in {ms:.0f} ms ({utter_s:.1f}s){' -> release' if act else ''}")
            if not act or self.pending_vad_transcript.get(client_uid):
                return
            # The speculation for this pause starts a few ms after it; wait for it.
            spec = None
            for _ in range(30):
                if self.speech_epoch.get(client_uid, 0) != epoch:
                    return  # the speaker went on
                spec = self.speculations.get(client_uid)
                if spec is not None and spec.epoch == epoch:
                    break
                spec = None
                await asyncio.sleep(0.01)
            if spec is None or spec.gate.is_open:
                return
            prev_task = self.current_conversation_tasks.get(client_uid)
            reply_running = prev_task is not None and not prev_task.done() and spec.task is not prev_task
            last_ts, last_text = self.last_vad_turn.get(client_uid, (0.0, ""))
            irq = self.interrupted_at.get(client_uid, 0.0)
            if (reply_running or (irq and irq > last_ts and time.monotonic() - irq < _TURN_MERGE_S)) and \
                    time.monotonic() - last_ts < _TURN_MERGE_S and last_text:
                return  # this turn merges with the previous one: the endpoint path handles it
            if face_detect.ADDRESSEE or self._omni_judged(client_uid):
                addressed, why = await self._addressed_settled(client_uid)
                if addressed is False:
                    return
            if self.speech_epoch.get(client_uid, 0) != epoch:
                return
            turn_timing.start()
            if await self._release_spec(websocket, client_uid, spec, f"early (smart-turn P={p:.2f})", utter_s):
                self.early_released[client_uid] = epoch
        except Exception as e:  # noqa: BLE001 — the endpoint remains the fallback
            logger.debug(f"[smart-turn] check failed: {e!r}")

    def _omni_judged(self, client_uid: str) -> bool:
        """Whether the Thinker's camera look (omni_addressee) applies to this
        client: the render tab's session — the PROTO, whose camera the frames
        come from and whose visitor's voice this is. The judge is process-wide,
        so without this a laptop browser (no camera) speaking while the PROTO's
        camera showed an empty room was muted by it. While the test hook
        injects (/render-stream/mic-inject, scripts/omni_probe.py --cam) its
        frames stand in for the camera for every client, so the unattended
        probe — a plain client — exercises the judge."""
        return omni_addressee.ENABLED and (
            client_uid in self.render_clients or render_hub.injecting
        )

    async def _speech_onset_actions(self, websocket: WebSocket, aside: bool) -> None:
        """What a speech onset triggers, unless the speech is an aside (visitors
        talking among themselves while she speaks: judged by the camera)."""
        # Free PC-B GPU1 for THIS turn's ASR + TTS: the previous reply's
        # in-flight motion generations are obsolete the moment the user
        # starts talking, but were measured still draining (queued
        # +4.3s) — slowing whisper (asr 250 -> 1546ms) and GSV's first
        # chunk (500 -> ~700ms). Cancelling closes their gateway
        # streams, which aborts the service-side queue within ~0.2s.
        try:
            from . import motion_cache  # lazy: optional subsystem

            if not aside:
                motion_cache.cancel_all()
        except Exception:  # noqa: BLE001 — never break the VAD path
            pass
        # The barge-in signal. This VAD has the proximity gate and decides
        # what becomes a turn; the client's has none, and when it decided
        # barge-in it stopped her for voices this one then ignored — she
        # went quiet and nothing answered (2026-10-23 21:57:47, 22:09:17,
        # 22:37:59). The client interrupts a reply on this instead.
        if not aside:
            try:
                await websocket.send_text(
                    json.dumps({"type": "control", "text": "user-speech-start"})
                )
            except Exception:  # noqa: BLE001 — client may be gone
                pass

    async def _speech_onset_omni(
        self, websocket: WebSocket, client_uid: str, t0: float, yunet_aside: bool
    ) -> None:
        """Onset actions after the Thinker's look at the frame (bounded wait):
        the look decides the aside; the YuNet rule only when it has no verdict."""
        addressed, why = await omni_addressee.JUDGE.settle(t0, timeout=0.25)
        aside = (addressed is False) if addressed is not None else yunet_aside
        self.barge_held.pop(client_uid, None)
        _busy, _over = turn_contingency.busy()
        _busy = _busy or turn_contingency.busy_room()  # measured within the last minutes (her first words of a reply tell nothing yet)
        if _busy and self._she_speaks(client_uid):
            # 2026-11-13, a school visit: of 68 turns she took, 44 replies were cut
            # off within a second by the next voice in the room ("the visitor goes
            # on"), and the child asked "Do you like marshmallows?" three times.
            # With voices over her all the time, one more is no interruption.
            aside = True
            logger.info(f"[addressee] onset while she speaks in a busy room (voices over {'her speech lately' if _over is None else format(_over, '.0%') + ' of her speech'}): she finishes")
        elif self._is_continuation(client_uid, t0):
            # The visitor she just took a turn from is going on (see _CONT_S):
            # stop now, whatever a look at the frame said.
            aside = False
            logger.info("[addressee] onset right after her turn: the visitor goes on — she stops")
        elif aside:
            logger.info(f"[addressee] onset: not to her ({why}) — she keeps talking")
        elif _BARGE_CONFIRM and self._she_speaks(client_uid):
            # A face in front is not a voice for her: beside the TV, the person
            # watching it faces her too, and the TV cut her off mid-reply again
            # and again (2026-11-06 12:41-12:44, 14 times in 3 minutes). She
            # stops once the speaker is seen (active_speaker) — at 0.7 s of
            # speech (_barge_probe), the first pause (_speculate) or the endpoint.
            self.barge_held[client_uid] = t0
            logger.info(f"[addressee] onset while she speaks ({why}): she goes on until the words are heard")
            _spawn(self._barge_probe(websocket, client_uid, t0))
            aside = True
        await self._speech_onset_actions(websocket, aside)

    def _she_speaks(self, client_uid: str) -> bool:
        """Is her voice playing now, or a released reply being made? (A
        speculation still behind its gate has said nothing yet.)"""
        if client_uid in self.render_clients and render_hub.speaking:
            return True
        task = self.current_conversation_tasks.get(client_uid)
        if task is None or task.done():
            return False
        spec = self.speculations.get(client_uid)
        return not (spec is not None and spec.task is task and not spec.gate.is_open)

    async def _barge_probe(self, websocket: WebSocket, client_uid: str, t0: float) -> None:
        """A held barge-in, judged on the first _BARGE_PROBE_S of speech instead
        of waiting for a pause: a visitor talking over her (1.8 s without a
        pause, 2026-11-06 13:15:46) had her talk on for 1.7 s. Undecided here
        (the words of a fragment are often "not to her") = the pause / the
        endpoint still decide."""
        try:
            await asyncio.sleep(max(0.0, t0 + _BARGE_PROBE_S - time.monotonic()))
            if self.barge_held.get(client_uid) != t0:
                return
            audio = self._utterance_audio_b64(client_uid)
            if not audio:
                return
            asd = self._asd_check(audio, time.monotonic(), "barge probe")
            ok, why = await self._addressee_policy(client_uid, None, asd, ears_audio=audio)
            if self.barge_held.get(client_uid) != t0:
                return
            if ok and self.barge_held.get(client_uid) == t0:
                await self._confirm_barge(websocket, client_uid, f"{why}; probed at {_BARGE_PROBE_S:.1f} s")
        except Exception as e:  # noqa: BLE001 — the pause / endpoint still decide
            logger.debug(f"[addressee] barge probe failed: {e!r}")

    async def _confirm_barge(self, websocket: WebSocket, client_uid: str, why: str) -> None:
        t0 = self.barge_held.pop(client_uid, None)
        if t0 is None:
            return
        logger.info(f"[addressee] barge-in confirmed {(time.monotonic() - t0) * 1000:.0f} ms after the onset ({why})")
        await self._speech_onset_actions(websocket, False)

    def _drop_barge(self, client_uid: str, why: str) -> None:
        if self.barge_held.pop(client_uid, None) is not None:
            logger.info(f"[addressee] no barge-in: she kept talking ({why})")

    def _addressed_now(self, client_uid: str) -> tuple[bool | None, str]:
        """Was the current utterance addressed to her, from what is known NOW
        (no waiting): the Thinker's looks first, then the YuNet rule."""
        t0, t1 = self.turn_started.get(client_uid, time.monotonic()), time.monotonic()
        if self._omni_judged(client_uid):
            v = omni_addressee.JUDGE.judge(t0, t1)
            if v[0] is not None:
                return v
        if face_detect.ADDRESSEE:
            return face_detect.ATTENTION.judge(t0, t1)
        return None, "no addressee check"

    async def _addressed_settled(self, client_uid: str) -> tuple[bool | None, str]:
        """_addressed_now for the endpoint: waits (bounded) for a look in flight."""
        t0 = self.turn_started.get(client_uid, time.monotonic())
        if self._omni_judged(client_uid):
            v = await omni_addressee.JUDGE.settle(t0, timeout=0.3)
            if v[0] is not None:
                return v
        if face_detect.ADDRESSEE:
            return face_detect.ATTENTION.judge(t0, time.monotonic())
        return None, "no addressee check"

    def _watch_reply(self, client_uid: str, task) -> None:
        """Note when this reply task ends, and the reply it stored (for _reply_context)."""
        if task is None:
            return

        def _done(t) -> None:
            text = ""
            try:
                ctx = self.client_contexts.get(client_uid)
                if ctx is not None and ctx.history_uid and not t.cancelled():
                    hist = get_history(ctx.character_config.conf_uid, ctx.history_uid)
                    if hist and hist[-1].get("role") == "ai":
                        text = str(hist[-1].get("content", ""))
            except Exception:  # noqa: BLE001
                pass
            self.last_reply[client_uid] = (time.monotonic(), text)
            if client_uid in self.render_clients:
                _she = voice_direction.she_line()
                if _she:
                    logger.info(f"[direction] her own voice — {_she}")

        task.add_done_callback(_done)

    _FOLLOWUP_S = float(os.getenv("ADDRESSEE_FOLLOWUP_S", "6.0"))

    def _reply_context(self, client_uid: str, onset: float | None = None) -> str | None:
        """What she just said, for the words check, when this utterance follows it
        (within ADDRESSEE_FOLLOWUP_S of her reply's end, or while she speaks):
        "Yes." / "Why?" / "Thank you." are answers to her only after something."""
        onset = self.turn_started.get(client_uid, time.monotonic()) if onset is None else onset
        end_t, said = self.last_reply.get(client_uid, (0.0, ""))
        if said and end_t and (onset - end_t <= self._FOLLOWUP_S or self._she_speaks(client_uid)):
            return said[-300:]
        return None

    def _mark_accepted(self, client_uid: str) -> None:
        """She took this turn (released a reply, or holds it for more): the
        visitor going on right after it is continuing it (_CONT_S)."""
        self.accepted_at[client_uid] = self.turn_end_at.get(client_uid, time.monotonic())

    def _is_continuation(self, client_uid: str, onset: float | None = None) -> bool:
        """Speech that began within _CONT_S of the end of a turn she took, or
        while an unfinished turn waits for its second half."""
        if self.pending_vad_transcript.get(client_uid):
            return True
        onset = self.turn_started.get(client_uid, time.monotonic()) if onset is None else onset
        acc = self.accepted_at.get(client_uid)
        return acc is not None and 0 <= onset - acc <= _CONT_S

    # ---- Her name opens a conversation (2026-11-09 04:17, the user's choice) ----
    # With someone standing in front of her, everything the one microphone hears
    # was theirs: a robot talking beside the PROTO, the TV, the visitor talking to
    # a friend. Neither loudness, voice features nor lips told those apart on the
    # PROTO. So, on top of the camera rule (someone stopped close, facing her),
    # a conversation starts only when her name is heard ("Iris, ..."), as with
    # Alexa or Google. While it is open, the visitor goes on without the name.
    # It ends when (a) nothing was said to her for WAKE_FOLLOW_S after her reply,
    # (b) nobody was near for 3 s (they walked off), or (c) they said goodbye.
    # WAKE_NAME=0 turns this off (the camera rule alone, as before).
    _WAKE = os.getenv("WAKE_NAME", "0") == "1"
    _WAKE_FOLLOW_S = float(os.getenv("WAKE_FOLLOW_S", "10") or 10)
    # Her name as Whisper and the Thinker write it from the PROTO's 4 kHz
    # microphone (Whisper's vocabulary hint has "Iris").
    _NAME_RE = re.compile(r"\b(?:iris|irish|iriss|irises|iriz|irus|iras|irys|eyeris|ayris|ires)\b", re.IGNORECASE)
    _BYE_RE = re.compile(
        r"\b(?:bye|goodbye|good-bye|see you|see ya|that's all|that is all|that's it for now|have a (?:good|nice) (?:day|one|night))\b",
        re.IGNORECASE,
    )
    _INVITE = "If you are talking to me, please start with my name, Iris."
    _INVITE_GAP_S = 60.0

    def _session_gate(self, client_uid: str, texts: list) -> tuple[bool, str]:
        """(ok, why) for a turn the camera rule already took as hers."""
        if not self._WAKE or client_uid not in self.render_clients:
            return True, ""
        texts = [t for t in texts if t]
        onset = self.turn_started.get(client_uid, time.monotonic())
        if any(self._NAME_RE.search(t) for t in texts):
            self.session_open[client_uid] = True
            why = "her name was said"
        elif self.session_open.get(client_uid):
            end_t, _ = self.last_reply.get(client_uid, (0.0, ""))
            last = max(end_t, self.accepted_at.get(client_uid, 0.0))
            if face_detect.ATTENTION.left_between(last, onset, 3.0):
                self.session_open[client_uid] = False
                return False, "the conversation ended (they walked off) and her name was not said"
            if not (self._she_speaks(client_uid) or onset - last <= self._WAKE_FOLLOW_S):
                self.session_open[client_uid] = False
                return False, (f"the conversation ended (nothing said to her for {onset - last:.0f} s) "
                               "and her name was not said")
            why = "the conversation is open"
        else:
            return False, "her name was not said, and no conversation is open"
        if any(self._BYE_RE.search(t) for t in texts):
            self.session_open[client_uid] = False
            why += "; goodbye — the conversation ends after this reply"
        return True, why

    async def _maybe_invite(self, websocket: WebSocket, client_uid: str) -> None:
        """Someone stopped in front and talked without her name: tell them how to
        start, once a minute at most."""
        now = time.monotonic()
        if now - self.invite_at.get(client_uid, -1e9) < self._INVITE_GAP_S or self._she_speaks(client_uid):
            return
        self.invite_at[client_uid] = now
        context = self.client_contexts.get(client_uid)
        if context is None:
            return
        try:
            from .utils import backchannel

            key = context.character_config.conf_uid
            clip = self._invite_clip.get(key)
            if clip is None:
                clip = base64.b64encode(backchannel._trim_wav(await backchannel._synthesize(context, self._INVITE))).decode()
                self._invite_clip[key] = clip
            await websocket.send_text(json.dumps({"type": "backchannel", "audio": clip}))
            logger.info(f"[wake] invited the visitor to say her name: {self._INVITE!r}")
        except Exception as e:  # noqa: BLE001 — never break a turn over this
            logger.debug(f"[wake] invite skipped: {e!r}")

    # ---- A talker that was already there, and nobody's mouth moving ----------
    # (2026-11-09) With a visitor standing in front, the robot next to the PROTO
    # or the TV was answered as if the visitor had spoken. No single cue tells
    # them apart on the PROTO, so a turn the camera rule took is dropped only when
    # THREE agree:
    #   1. a voice was already running before it began (>= BG_BEFORE of the 6 s),
    #   2. it did not come in louder than that voice (step <= BG_STEP_DB),
    #   3. the face in front shows no talking over >= 1 s of voiced picture
    #      (lip score <= BG_LIP with <= 10 % of frames > 0).
    # A visitor speaking in a quiet moment fails 1; one speaking up over the
    # loudspeaker fails 2; one whose lips are seen moving fails 3; short
    # utterances are never dropped by this (3 needs a second of picture).
    # BG_TALKER_DROP=0 logs the verdict without dropping.
    _BG_DROP = os.getenv("BG_TALKER_DROP", "1") == "1"
    _BG_BEFORE = float(os.getenv("BG_BEFORE", "0.4"))
    _BG_STEP_DB = float(os.getenv("BG_STEP_DB", "2.0"))
    _BG_LIP = float(os.getenv("BG_LIP", "-0.6"))

    _VISITOR_RESET = os.getenv("VISITOR_RESET", "1") == "1"

    async def _maybe_new_visitor(self, websocket: WebSocket, client_uid: str) -> None:
        """A new visitor gets a new conversation. She stands in a public place:
        one history ran from 11-04 to 11-09 (660 messages, hundreds of visitors),
        and its stock replies were the model's examples — "Hello. I am Iris, the
        android guide for this maker space." answered "How are you?", "Thank
        you." and "Can you make a guide for me?" alike (19:51), though the same
        questions with no history are answered properly. When someone walks up
        after the place in front of her stood empty (face_detect: nobody within
        NEAR_M for VISITOR_GAP_S) since the last turn she took, the old thread
        is left in its file and a fresh one starts. VISITOR_RESET=0 turns it off."""
        if not self._VISITOR_RESET or client_uid not in self.render_clients:
            return
        last = self.accepted_at.get(client_uid)
        # After a (re)start the thread resumed is whatever ran last — other
        # visitors' talk: the first voice heard starts a fresh one, once.
        first = client_uid not in self._visitor_fresh
        if not first and (last is None or face_detect.ATTENTION.arrived_at <= last):
            return
        # ...and the talk had stopped: a face lost for a while in the middle of
        # an exchange (stepped close under the camera, looked down at a phone,
        # the detector missing it) coming back is the same person, not a new one.
        _quiet_from = max(last or 0.0, self.last_reply.get(client_uid, (0.0, ""))[0])
        if not first and time.monotonic() - _quiet_from < face_detect.ATTENTION.ARRIVE_GAP_S:
            return
        task = self.current_conversation_tasks.get(client_uid)
        if (task is not None and not task.done()) or self._she_speaks(client_uid):
            return
        context = self.client_contexts.get(client_uid)
        if context is None or context.agent_engine is None:
            return
        try:
            conf_uid = context.character_config.conf_uid
            # Nothing was said in the current thread (the last arrival only
            # walked past, or a voice from the room set this off): it is as
            # fresh as a new one — no empty file per passer-by.
            if not first and context.history_uid and not any(
                m.get("role") in ("human", "ai") for m in (get_history(conf_uid, context.history_uid) or [])
            ):
                self.accepted_at.pop(client_uid, None)
                return
            history_uid = create_new_history(conf_uid)
            if not history_uid:
                return
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(conf_uid=conf_uid, history_uid=history_uid)
            self._visitor_fresh.add(client_uid)
            self.accepted_at.pop(client_uid, None)
            self.session_open.pop(client_uid, None)
            why = "first voice since this session began" if last is None else f"someone new walked up {time.monotonic() - last:.0f} s after her last turn"
            logger.info(f"[visitor] {why} — new conversation {history_uid}")
            await self._send_quiet(websocket, json.dumps({"type": "new-history-created", "history_uid": history_uid}))
        except Exception as e:  # noqa: BLE001 — the old thread simply goes on
            logger.warning(f"[visitor] could not start a new conversation: {e!r}")

    _HOLD_QUIET_S = float(os.getenv("VAD_UNFINISHED_QUIET_S", "1.0") or 0)

    async def _hold_wait(self, client_uid: str, hold_s: float) -> None:
        """The wait for the rest of an unfinished sentence. It was a fixed
        sleep: 1.5 s after the transcript, so a visitor who simply stopped
        there ("So should human be afraid of") heard her ~2.1 s after their last
        word. With the two microphones (voice_direction) the wait ends as soon
        as no voice has come from in front for _HOLD_QUIET_S since they spoke —
        a pause inside a sentence is shorter than that, and a restart is heard
        here before it is loud enough to open a turn. Without the stereo
        stream, or with a short hold, it is the fixed sleep as before."""
        t_end = time.monotonic() + hold_s
        if not self._HOLD_QUIET_S or hold_s <= self._HOLD_QUIET_S:
            await asyncio.sleep(hold_s)
            return
        t_on = self.turn_started.get(client_uid, time.monotonic() - 5.0)
        while (now := time.monotonic()) < t_end:
            alive, last = voice_direction.last_front_voice(t_on, self.turn_end_at.get(client_uid))
            if not alive:
                await asyncio.sleep(t_end - now)
                return
            quiet_since = last if last is not None else self.turn_end_at.get(client_uid, now) - 0.3
            if now - quiet_since >= self._HOLD_QUIET_S:
                logger.info(f"[server-vad] hold ended early: no voice from in front for {now - quiet_since:.2f} s "
                            f"({t_end - now:.2f} s of the {hold_s:g} s hold left)")
                return
            await asyncio.sleep(0.05)

    _DIR_DECIDES = os.getenv("DIRECTION_DECIDES", "1") == "1"
    _DIR_SIDE = float(os.getenv("DIRECTION_SIDE", "0.3"))    # at most this share from in front: not the visitor
    _DIR_CENTRE = float(os.getenv("DIRECTION_FACE_CENTRE", "0.15"))  # how far from the picture's middle the face may be
    _DIR_FRONT = float(os.getenv("DIRECTION_FRONT", "0.6"))  # at least this: the visitor, whatever the lips say

    async def _bg_talker(self, client_uid: str, turn_audio: str | None) -> tuple[bool, str]:
        """(drop, why). Costs the lip check (0.1-0.5 s) only when 1 and 2 hold."""
        if not turn_audio or client_uid not in self.render_clients:
            return False, ""
        t_on = self.turn_started.get(client_uid, time.monotonic())
        t_end = self.turn_end_at.get(client_uid, time.monotonic())
        # Said over her reply in a busy room: not an interruption she can tell
        # from the room (see _speech_onset_omni). Taking it would start a new
        # reply and cancel the one she is in.
        if turn_contingency.busy_room():
            _barge = turn_contingency.features(t_on, t_end)["barge"]
            if _barge is not None and _barge >= 0.5:
                return True, f"spoken over her reply ({_barge:.0%} of it) in a busy room"
        # Two microphones (the small PROTO, voice_direction.py): where the voice
        # came from settles it, ahead of the timing-and-lips rule below — on
        # 2026-11-09 19:44 that rule dropped a visitor's question spoken from
        # straight ahead (lips -0.82) and the camera rule answered a sound from
        # the side while he stood there. His ten turns: 92-100% of the voiced
        # blocks in front; the room's: 0-13%. Her own voice is left out of it.
        d = voice_direction.features(t_on, t_end) if self._DIR_DECIDES else None
        if d is not None and d["n"] >= 2:
            said = (f"{d['front_w']:.0%} of the voice came from in front ({d['n']} blocks, "
                    f"median lag {d['median']:+.1f}, {d['db']:.0f} dBFS)")
            # "From the side" is only told against a face in the MIDDLE of the
            # picture: someone standing a little to one side of her speaks from
            # there too (20 cm off at 0.45 m is already ~5 samples), and how
            # picture position maps to lag is not calibrated yet.
            fx = face_detect.ATTENTION.face_x(t_on - 1.0, t_end)
            said += f", face at {fx:.2f} across the picture" if fx is not None else ", face position unknown"
            if d["front_w"] <= self._DIR_SIDE and fx is not None and abs(fx - 0.5) <= self._DIR_CENTRE:
                return True, "the voice came from the side, not from the person in front: " + said
            if d["front_w"] <= self._DIR_SIDE:
                logger.info(f"[direction] side voice, but the face is off-centre or unseen — left to the other checks: {said}")
            if d["front_w"] >= self._DIR_FRONT:
                return False, said
        if not lip_motion.ENABLED:
            return False, ""
        x = turn_contingency.features(t_on, t_end)
        if x["before"] is None or x["before"] < self._BG_BEFORE:
            return False, ""
        if x["step"] is not None and x["step"] > self._BG_STEP_DB:
            return False, f"a voice was running before ({x['before']:.0%}) but this one came in {x['step']:+.1f} dB louder"
        try:
            best, _ = await asyncio.wait_for(asyncio.to_thread(lip_motion.best_b64, turn_audio, t_end), 1.5)
        except Exception as e:  # noqa: BLE001 — no picture verdict: answer
            return False, f"lip check failed ({e!r})"
        step = "n/a" if x["step"] is None else f"{x['step']:+.1f} dB"
        if best is None or best[2] < 25:
            return False, f"a voice was running before ({x['before']:.0%}, step {step}); too little picture to tell"
        why = (f"a voice was already running ({x['before']:.0%} of the 6 s before, step {step}) and the face in front "
               f"{'is not' if best[0] <= self._BG_LIP and best[1] <= 0.10 else 'is'} talking (lips {best[0]:+.2f}, {best[1]:.0%} of {best[2]} frames)")
        return best[0] <= self._BG_LIP and best[1] <= 0.10, why

    _NOD_GAP_S = float(os.getenv("LISTEN_NOD_GAP_S", "2.5") or 2.5)

    def _maybe_listening_nod(self, websocket: WebSocket, client_uid: str) -> None:
        """At a pause in the visitor's speech (or its end), tell the avatar to
        nod: what they see first after they stop is that she took it in — ~0.3-0.5 s
        before her voice can start. Not while she speaks, not to someone who is
        turned away or not there, not more than once per _NOD_GAP_S, and not
        for a sound under 0.6 s. LISTEN_NOD=0 turns it off."""
        if os.getenv("LISTEN_NOD", "1") == "0":
            return
        now = time.monotonic()
        onset = self.turn_started.get(client_uid, now)
        if now - onset < 0.6 or now - self._last_nod.get(client_uid, 0.0) < self._NOD_GAP_S:
            return
        if self._she_speaks(client_uid):
            return
        who = face_detect.ATTENTION.presence(onset, now)[0]
        if who in ("away", "nobody"):
            return
        self._last_nod[client_uid] = now
        _spawn(self._send_quiet(websocket, json.dumps({"type": "control", "text": "listening-nod"})))

    @staticmethod
    async def _send_quiet(websocket: WebSocket, text: str) -> None:
        try:
            await websocket.send_text(text)
        except Exception:  # noqa: BLE001
            pass

    async def _asd_prepare(self, client_uid: str) -> None:
        """While this utterance lasts, keep the lip model's visual half up to
        date (active_speaker.prepare), so the verdict at the pause is quick."""
        try:
            while self.in_utterance.get(client_uid):
                await asyncio.to_thread(active_speaker.prepare)
                await asyncio.sleep(0.5)
        except Exception as e:  # noqa: BLE001 — the verdict then does it all itself
            logger.debug(f"[asd] prepare failed: {e!r}")

    async def _understood_in_busy_room(self, turn_audio: str, whisper_text: str) -> tuple[bool, str]:
        """In a busy room, were words really heard? Whisper and the Thinker both
        write SOMETHING for babble — on 2026-11-13 Whisper wrote "Thank you for
        watching."-style lines for 48 % of the pauses and the Thinker "I'm going
        to go to the store." 108 times — and she answered those ("The restroom is
        down the hall to your left."). They are two different models: of 527
        such pairs 6 agreed, while the pairs that did agree were real speech.
        Only where the room is busy: with the 4 kHz microphone and an accented
        visitor in a quiet room, half the REAL turns disagreed (2026-11-09)."""
        heard = ""
        try:
            heard = await omni_addressee.HEARD.get(turn_audio, 1.2) or ""
        except Exception:  # noqa: BLE001
            heard = ""
        if not heard:
            return True, "busy room, no second transcript to compare"
        a = turn_contingency.agreement(heard, whisper_text)
        said = f"the two transcripts agree {a:.0%}: {heard[:50]!r} / {whisper_text[:50]!r}"
        if a >= _UNDERSTOOD_MIN:
            return True, "busy room, " + said
        return False, "busy room and no words both transcripts heard — " + said

    def _start_heard(self, audio: str | None, why: str):
        """The Thinker's transcription of this audio (shared with the reply's,
        omni_addressee.HEARD) and whether its words stop mid-sentence, started
        now: a task -> (unfinished | None, heard). None without audio."""
        if not (omni_addressee.HEARD_ON and audio):
            return None
        return asyncio.create_task(omni_addressee.COMPLETE.judge_audio(audio, why))

    def _asd_check(self, wav_b64: str | None, end_t: float, why: str):
        """Who is talking (active_speaker.speaking) for this audio, which ends at
        monotonic end_t: a task -> (verdict, why). The last 3 s are enough to see
        whose lips move, and keep it to ~100-150 ms of CPU, off the loop."""
        if not (active_speaker.ENABLED and wav_b64):
            return None

        def run():
            pcm = np.frombuffer(base64.b64decode(wav_b64)[44:], dtype=np.int16).astype(np.float32) / 32768.0
            pcm = pcm[-int(3.0 * 16000):]
            return active_speaker.speaking(end_t - len(pcm) / 16000, end_t, pcm, why)

        return asyncio.create_task(asyncio.to_thread(run))

    @staticmethod
    async def _spec_heard(spec, timeout: float = 0.3) -> str:
        """What the Thinker heard for this speculation's audio ("" if not known yet)."""
        fut = getattr(spec, "heard", None) if spec is not None else None
        if fut is None:
            return ""
        try:
            return (await asyncio.wait_for(asyncio.shield(fut), timeout))[1] or ""
        except Exception:  # noqa: BLE001
            return ""

    async def _unfinished(self, *texts: str, spec=None, votes: list | None = None) -> bool:
        """Does any transcript of this turn stop mid-sentence? The Thinker judges
        the grammar (omni_addressee.COMPLETE — any speaker, any topic); its
        verdict on the speculation's transcript is already under way at the
        pause. Without a verdict, the old ending rules decide."""
        checks = []
        fut = getattr(spec, "heard", None) if spec is not None else None
        heard = ""
        if fut is not None:
            try:
                u, heard = await asyncio.wait_for(asyncio.shield(fut), 0.4)
                checks.append((heard, u))
            except Exception:  # noqa: BLE001
                pass
        for t in texts:
            t = (t or "").strip()
            if t and t != (heard or "").strip():
                checks.append((t, await omni_addressee.COMPLETE.unfinished(t, "whisper") if _OMNI_AUDIO_IN else None))
        v = [bool(u if u is not None else _transcript_looks_incomplete(t)) for t, u in checks if t]
        if votes is not None:
            votes.extend(v)
        return any(v)

    async def _addressee_policy(
        self, client_uid: str, ears: int | None, asd=None, ears_audio: str | None = None
    ) -> tuple[bool, str]:
        """Was this speech for her? Decided by WHO is talking, never by what was
        said (2026-11-08 — see active_speaker.py for why the words could not):

          a face in front moved its lips with the voice, and faces her -> to her
          ... and is turned to someone else                           -> not to her
          faces in front seen well, none of them spoke                -> not to her
            (the TV, a phone, someone out of view)
          not seen well enough (small/no face, no camera):
            someone near facing her                                   -> to her
            otherwise the ears (omni_addressee.AUDIO, 0-9; none -> to her)
        """
        onset = self.turn_started.get(client_uid, time.monotonic())
        who, fwhy = face_detect.ATTENTION.presence(onset, time.monotonic())
        v, awhy, facing = None, "", None
        if asd is not None and os.getenv("ASD_DROP", "0") != "1":
            # Not yet trusted on the PROTO (only logged): never wait for it.
            asd.add_done_callback(
                lambda f: logger.info(f"[asd] (log only) {f.result()[0]}: {f.result()[1][:200]}")
                if not f.cancelled() and f.exception() is None else None
            )
            asd = None
        if asd is not None:
            try:
                v, awhy, facing = await asyncio.wait_for(asyncio.shield(asd), 1.0)
            except Exception as e:  # noqa: BLE001 — not seen: the rules below
                v, awhy = None, f"asd: {e!r}"
        if v == "speaking":
            if (facing or 0) >= face_detect.Attention.MIN_FRAC or who == "facing":
                return True, f"speaker seen, facing her — {awhy}"
            return False, f"speaker seen, turned away from her — {awhy}"
        if v == "silent" and os.getenv("ASD_DROP", "0") == "1":
            return False, f"voice not from anyone in front — {awhy}"
        # (2026-11-09, first test on the PROTO: the visitor talking straight at
        # her, 213-220 px face crops, came out -0.90/-1.11 "silent" and was
        # dropped. Until the lip verdict is right on the PROTO it only logs;
        # ASD_DROP=1 lets it drop turns again.)
        if who is not None and os.getenv("ADDRESSEE_ENGAGED", "1") == "1":
            eng, ewhy = face_detect.ATTENTION.engaged(onset, time.monotonic())
            if eng is not None:
                return eng, ewhy
        if who == "facing":
            etxt = f"ears {ears}/9" if ears is not None else "ears not asked"
            return True, f"{etxt}, someone near facing her ({fwhy}; {awhy or 'asd: none'})"
        # Only here are the ears needed. They are asked now rather than at every
        # pause: each ask is one more audio request on the Thinker's GPU, beside
        # the reply being made and the transcript (2026-11-08).
        if ears is None and ears_audio and omni_addressee.AUDIO_ENABLED:
            try:
                ears = await asyncio.wait_for(omni_addressee.AUDIO.score(ears_audio, "endpoint"), 1.0)
            except Exception:  # noqa: BLE001 — no verdict: answer
                ears = None
        if who is not None and os.getenv("ADDRESSEE_ENGAGED", "1") == "1":
            # The camera decides: only someone who stopped close and faces her
            # (face_detect.Attention.engaged). The ears are not asked.
            eng, ewhy = face_detect.ATTENTION.engaged(onset, time.monotonic())
            if eng is not None:
                return eng, ewhy
        loud = ears is None or ears >= omni_addressee.AUDIO_MIN
        etxt = f"ears {ears}/9" if ears is not None else "ears: none"
        return loud, f"{etxt}, {who or 'no camera'} ({fwhy}; {awhy or 'asd: none'})"

    async def _ears_verdict(
        self, client_uid: str, spec, turn_audio: str | None, asd=None
    ) -> tuple[bool | None, str]:
        """Was this turn said to her: the ears (the verdict the speculation
        started at its pause, else one on the turn's audio) and who is talking
        (`asd`, else the speculation's), through _addressee_policy. (None, why)
        when nothing could be judged — the caller then falls back to the camera."""
        if not omni_addressee.AUDIO_ENABLED and not active_speaker.ENABLED:
            return None, "ears off"
        heard = None
        try:
            if spec is not None and spec.ears is not None:
                heard = await asyncio.wait_for(asyncio.shield(spec.ears), 0.5)
        except Exception:  # noqa: BLE001 — timeout / failure: asked again if needed
            heard = None
        if asd is None and spec is not None:
            asd = spec.asd
        if asd is None and turn_audio:
            asd = self._asd_check(turn_audio, self.turn_end_at.get(client_uid, time.monotonic()), "endpoint")
        ears_audio = turn_audio if turn_audio else (getattr(spec, "audio", None) if spec is not None else None)
        if lip_motion.ENABLED and ears_audio and client_uid in self.render_clients:
            # log only: how the visitor's mouth moved (lip_motion.py)
            _end = self.turn_end_at.get(client_uid, time.monotonic()) if turn_audio else (
                spec.started if spec is not None else time.monotonic()
            )
            _fut = asyncio.ensure_future(asyncio.to_thread(lip_motion.check_b64, ears_audio, _end))
            _fut.add_done_callback(
                lambda f: logger.info(f"[lip] {f.result()}") if not f.cancelled() and f.exception() is None
                else logger.debug(f"[lip] failed: {f.exception()!r}")
            )
        if heard is None and asd is None and not ears_audio:
            return None, "ears: no verdict"
        return await self._addressee_policy(client_uid, heard, asd, ears_audio=ears_audio)

    def _utterance_audio_b64(self, client_uid: str) -> str | None:
        """omni mode: the utterance so far as a base64 16 kHz mono WAV for the
        speech-in LLM — the same audio the cumulative ASR reads (the mic ring
        from the utterance start; the viewer's original mic for the render tab).
        None in cascade mode or when nothing is being said."""
        if not _OMNI_AUDIO_IN:
            return None
        return self._audio_b64_from(client_uid, self.utterance_start.get(client_uid))

    def _audio_b64_from(self, client_uid: str, start: int | None) -> str | None:
        """The mic ring from `start` (a ring index) to now as a base64 16 kHz
        WAV; the viewer's original mic for the render tab (see above)."""
        audio = self._audio_np_from(client_uid, start)
        return None if audio is None else _wav_b64_16k(audio)

    def _audio_np_from(self, client_uid: str, start: int | None) -> np.ndarray | None:
        """_audio_b64_from's audio, as 16 kHz float32 (None under 0.1 s)."""
        ring = self.mic_rings.get(client_uid)
        if ring is None or start is None:
            return None
        start = max(start, ring.end - _ASR_MAX_UTTERANCE)
        audio = ring.slice(start, ring.end)
        if client_uid in self.render_clients and render_hub.viewer_mic_live():
            try:
                original = render_hub.viewer_audio_16k(
                    ring.wall(start) - _VIEWER_LEAD_S, time.time()
                )
                if len(original) >= len(audio) // 2:
                    audio = original
            except Exception as e:  # noqa: BLE001 — the tab's copy still works
                logger.debug(f"[omni-audio] viewer mic unavailable: {e!r}")
        if len(audio) < 1600:  # under 0.1 s: nothing to hear
            return None
        return audio

    async def _asr_one_at_a_time(self, context, call):
        """One Whisper request in flight per session. A pause's transcript
        supersedes the one before it, and that task is cancelled — but the
        request runs in a thread and cannot be recalled, so the "cancelled"
        ones went on using the GPU: 366 of 1481 on 2026-11-13. Whisper takes
        192 ms alone, 351 ms beside one other, 486 ms beside two, and the
        transcript then missed the endpoint (417 ms at the median in the room).
        Here a superseded request that has not left yet never leaves (it is
        cancelled while waiting its turn); one already at Whisper is waited out."""
        lock = self._asr_locks.get(id(context))
        if lock is None:
            lock = self._asr_locks[id(context)] = asyncio.Lock()
        _tq = time.monotonic()
        async with lock:
            self._asr_wait_ms = (time.monotonic() - _tq) * 1000
            fut = asyncio.ensure_future(call())
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError:
                while not fut.done():
                    try:
                        await asyncio.wait([fut])
                    except asyncio.CancelledError:
                        continue
                if not fut.cancelled():
                    fut.exception()  # retrieved: nobody else will
                raise

    async def _transcribe(self, context, audio: np.ndarray, viewer_span, fast: bool = False) -> str:
        """ASR for a segment pause. For the render tab (the PROTO's voice) the
        audio is taken from the viewer's original mic at the server over
        `viewer_span` (render_stream.hub.viewer_mic) instead of the tab's
        damaged copy; `audio` stays the fallback if that comes up short."""
        if viewer_span is not None:
            original = await asyncio.to_thread(render_hub.viewer_audio_16k, *viewer_span)
            if len(original) >= len(audio) // 2:
                audio = original
        name = asr_capture.segment((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        _t0 = time.monotonic()
        try:
            try:
                _quick = getattr(context.asr_engine, "async_transcribe_fast", None) if fast else None
                text = await self._asr_one_at_a_time(
                    context, lambda: _quick(audio) if _quick is not None else context.asr_engine.async_transcribe_np(audio)
                )
                if fast:
                    _lf = getattr(context.asr_engine, "last_fast", (0, 0, 0))
                    logger.info(
                        f"[asr] quick transcript in {(time.monotonic() - _t0) * 1000:.0f} ms ({len(audio) / 16000:.1f} s of audio): "
                        f"waited its turn {getattr(self, '_asr_wait_ms', 0):.0f} ms, wav {_lf[0]:.0f} ms, request {_lf[1]:.0f} ms (server {_lf[2]:.0f} ms)"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # Once more: a transient gateway hiccup should not cost the turn.
                # A second failure propagates; the finalize then falls back to the
                # previous pause's transcript, and the log says so.
                logger.warning(f"[server-vad] ASR failed ({e!r}); retrying once")
                text = await context.asr_engine.async_transcribe_np(audio)
        except asyncio.CancelledError:
            if name:
                asr_capture.transcript(name, "(cancelled)")
            raise
        if name:
            asr_capture.transcript(name, text)
        return text

    def _resume_latest_history(self, context: ServiceContext) -> None:
        """Attach the newest existing conversation for this character, or start
        a fresh one. Called on every connection so the thread survives reloads,
        restarts and character switches back and forth.

        Long threads are safe: the agent's own compaction (LLM_COMPACT_AT)
        summarises old turns in the background, and the per-request context trim
        is the final backstop before the model call."""
        try:
            # The system prompt every turn will carry (persona + this language's
            # directive, appended per turn by apply_client_language) — set BEFORE
            # the history load below schedules the prefix warm. Warmed without the
            # directive, the cache diverged at the end of the system prompt and
            # the first turn re-read the whole history: 2.8 s to the first
            # sentence on 2026-10-24 22:57, against 0.2-0.3 s from turn two.
            apply_client_language(context, getattr(context.character_config, "language", None))
            conf_uid = context.character_config.conf_uid
            histories = get_history_list(conf_uid)
            if histories:
                history_uid = histories[0]["uid"]
                context.history_uid = history_uid
                context.agent_engine.set_memory_from_history(conf_uid, history_uid)
                context.agent_engine.schedule_compaction_if_needed()
                logger.info(f"[history] resumed conversation {history_uid}")
            else:
                context.history_uid = create_new_history(conf_uid)
                logger.info(f"[history] started new conversation {context.history_uid}")
                # No history to load, but the system prompt alone is worth
                # prefilling (Rem's is ~12k tokens).
                context.agent_engine.schedule_prefix_warm()
            # The acknowledgment clips too, now rather than on the first turn
            # that wants one (utils/backchannel.prime).
            from .utils import backchannel  # lazy: optional

            backchannel.prime(context)
        except Exception as e:  # noqa: BLE001 — never block a connection over this
            logger.warning(f"[history] could not resume conversation: {e!r}")

    async def _release_dropped_turn(self, websocket: WebSocket) -> None:
        """Tell the client a turn ended even though we never started a reply.

        The frontend flips to 'thinking-speaking' the moment YOUR speech ends,
        and only leaves that state on conversation-chain-end. Every early
        `return` in _finalize_vad_turn used to skip that signal, so a dropped
        turn left the avatar frozen mid-"thinking" and the NEXT utterance was
        treated as an interrupt instead of a new turn — the user-visible
        "ミクが話してる途中に話すと、ちょくちょく返ってこない". Barge-in makes
        drops far more likely (the AI's own voice bleeds into the mic and
        transcribes as garbage), which is why it showed up there first.

        Not while a reply is running: chain-end carries no turn id, so the client
        took it as the END of that reply and its player finished at the next
        underrun — a pose reset mid-sentence and the rest of the reply played
        as a new turn."""
        uid = next((u for u, ws in self.client_connections.items() if ws is websocket), None)
        task = self.current_conversation_tasks.get(uid) if uid else None
        spec = self.speculations.get(uid) if uid else None
        # (This turn's own speculation is about to be rolled back; it is not a reply.)
        if task is not None and not task.done() and not (spec is not None and spec.task is task):
            return
        try:
            await websocket.send_text(
                json.dumps({"type": "control", "text": "conversation-chain-end"})
            )
        except Exception as e:  # noqa: BLE001 — client may already be gone
            logger.debug(f"[server-vad] could not release dropped turn: {e!r}")

    # --- speculative replies (see conversations/speculative.py) -------------

    def _spec_lock(self, client_uid: str) -> asyncio.Lock:
        return self.spec_locks.setdefault(client_uid, asyncio.Lock())

    async def _drop_speculation(
        self, client_uid: str, reason: str, epoch: int | None = None
    ) -> None:
        async with self._spec_lock(client_uid):
            spec = self.speculations.get(client_uid)
            if spec is None or (epoch is not None and spec.epoch != epoch):
                return
            self.speculations.pop(client_uid, None)
            await speculative.discard(spec, reason)

    def _worth_speculating(self, client_uid: str, text: str) -> bool:
        """The cheap half of _finalize_vad_turn's screening. Anything that turn
        would hold or drop is not worth a speculative LLM+TTS run; the endpoint
        stays authoritative either way (a miss is rolled back)."""
        t = text.strip()
        if not t or _is_junk_transcript(t) or _transcript_looks_incomplete(t):
            return False
        if len(t) <= 40 and any(h in t.lower() for h in _ASR_HALLUCINATIONS):
            return False
        context = self.client_contexts.get(client_uid)
        lang = getattr(context.character_config, "language", "ja") if context else "ja"
        return not (lang != "ja" and _CJK_CHARS_RE.search(t))

    async def _speculate(self, websocket: WebSocket, client_uid: str, tasks: list, epoch: int) -> None:
        """On a segment pause, start the reply to the transcript so far."""
        try:
            await self._speculate_inner(websocket, client_uid, tasks, epoch)
        except Exception as e:  # noqa: BLE001 — a failed speculation is just a miss
            logger.exception(f"[spec] speculation failed: {e!r}")

    async def _speculate_inner(self, websocket: WebSocket, client_uid: str, tasks: list, epoch: int) -> None:
        t0 = time.monotonic()
        # omni, audio-first: the Thinker hears the audio, so the reply can start on
        # the pause itself; the transcript (~0.43 s behind) only screens the run and
        # fills memory/history. Not when a held mid-thought transcript has to be
        # prepended — the audio from the utterance start does not cover it.
        spec_audio = None
        if _SPEC_AUDIO_FIRST and _OMNI_AUDIO_IN and not self.pending_vad_transcript.get(client_uid):
            spec_audio = self._utterance_audio_b64(client_uid)
        if spec_audio:
            spec = await self._start_speculation(
                websocket, client_uid, tasks, epoch, text=speculative.SPOKEN_PLACEHOLDER,
                asr_ms=(time.monotonic() - t0) * 1000, audio=spec_audio, placeholder=True,
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parts = [p for p in _turn_parts(results) if not _segment_is_hallucination(p)]
            text = "".join(parts).strip()
            if spec is None:
                return
            if spec.released:
                # Fast release already spoke on the audio; the words decide only
                # whether that was noise (retract) and what the panel/history say.
                t = text.strip()
                lang = getattr(self.client_contexts.get(client_uid).character_config, "language", "ja") if self.client_contexts.get(client_uid) else "ja"
                # Noise = empty / an ASR hallucination phrase / wrong script /
                # filler-only ("hmm", "uh"). NOT the word-count rule: the Thinker
                # heard the audio and chose to answer, and a one-word answer
                # ("Technology.") is a real turn in a kiosk conversation — it was
                # retracted as junk after a syllable (2026-11-04 07:58).
                _toks = [w.strip(".,!?。！？…-\"'").lower() for w in t.split()]
                _toks = [w for w in _toks if w]
                noise = (
                    not t
                    or (bool(_toks) and all(w in _JUNK_FILLERS and w not in _REAL_SHORT_WORDS for w in _toks))
                    or (len(t) <= 40 and any(h in t.lower() for h in _ASR_HALLUCINATIONS))
                    or (lang != "ja" and _CJK_CHARS_RE.search(t) is not None)
                )
                _why_noise = f"noise once transcribed: {t[-30:]!r}"
                if not noise and _UNDERSTOOD_GATE and client_uid in self.render_clients and turn_contingency.busy_room():
                    _h = await self._spec_heard(spec, 0.05)
                    _a = turn_contingency.agreement(_h, t) if _h else 1.0
                    if _a < _UNDERSTOOD_MIN:
                        noise = True
                        _why_noise = (f"busy room, the two transcripts agree {_a:.0%}: {_h[:40]!r} / {t[:40]!r}; "
                                      f"{render_hub.direct_chunks} audio chunks had gone to the viewer")
                if noise:
                    await speculative.discard(spec, f"retracted after fast release — {_why_noise}")
                    if client_uid in self.render_clients:
                        # The direct path has already queued the reply's audio at
                        # the viewer; cancelling the task does not reach it.
                        try:
                            await render_hub.viewer_flush("retracted")
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        await self._release_dropped_turn(websocket)
                    except Exception:  # noqa: BLE001
                        pass
                    return
                speculative.set_text(spec, text)
                # Keep the RELEASE time (set by _release_spec with no text yet):
                # an interrupt between release and this transcript must still
                # count as "after her turn" for the continuation merge — with
                # the transcript's own time it never did (large-v3 lands
                # 0.55-0.9 s after release).
                _prev = self.last_vad_turn.get(client_uid)
                _ts = _prev[0] if _prev is not None and not _prev[1] else time.monotonic()
                self.last_vad_turn[client_uid] = (_ts, text)
                logger.info(f"[spec] transcript caught up {(time.monotonic() - t0) * 1000:.0f} ms after the pause (released already): {text[:60]!r}")
                asr_capture.note(f"TURN {text}")
                try:
                    await websocket.send_text(json.dumps({"type": "user-input-transcription", "text": text}))
                except Exception:  # noqa: BLE001
                    pass
                return
            if not self._worth_speculating(client_uid, text):
                async with self._spec_lock(client_uid):
                    if self.speculations.get(client_uid) is spec:
                        self.speculations.pop(client_uid)
                        await speculative.discard(spec, f"not worth it once transcribed: {text[-30:]!r}")
                spec.text_ready.set()
                return
            speculative.set_text(spec, text)
            logger.info(
                f"[spec] transcript caught up {(time.monotonic() - t0) * 1000:.0f} ms "
                f"after the pause: {text[:60]!r}"
            )
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        asr_ms = (time.monotonic() - t0) * 1000
        parts = _turn_parts(results)
        parts = [p for p in parts if not _segment_is_hallucination(p)]
        text = (self.pending_vad_transcript.get(client_uid, "") + "".join(parts)).strip()
        if not self._worth_speculating(client_uid, text):
            logger.debug(f"[spec] skip: not worth it ({text[-30:]!r})")
            return
        await self._start_speculation(
            websocket, client_uid, tasks, epoch, text=text, asr_ms=asr_ms,
            audio=self._utterance_audio_b64(client_uid),
        )

    async def _start_speculation(
        self, websocket: WebSocket, client_uid: str, tasks: list, epoch: int, text: str,
        asr_ms: float, audio: str | None = None, placeholder: bool = False,
    ) -> speculative.Speculation | None:
        """Start the gated reply for `text` (+ `audio`); returns it, or None when
        this pause is not one to speculate on (group, WebRTC, stale, not to her)."""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group and len(group.members) > 1:
            return  # group replies broadcast; not gated
        if webrtc_manager.track_for_send(websocket.send_text) is not None:
            return  # WebRTC audio goes out on the track, around the gate
        if self.speech_epoch.get(client_uid, 0) != epoch:
            logger.debug("[spec] skip: the user resumed speaking during the ASR")
            return None
        if self._she_speaks(client_uid) and turn_contingency.busy_room():
            logger.info("[spec] skip: she is answering and the room is busy — she finishes first")
            return None
        # Starting a reply supersedes the running one (handle_conversation_trigger
        # cancels it), so a speculation on visitors talking among themselves
        # would cut her off mid-answer before the endpoint could ignore them.
        # Only while she is answering: with nothing running, speculating costs
        # nothing and keeps the reply fast if the turn turns out to be for her.
        running = self.current_conversation_tasks.get(client_uid)
        spec_now = self.speculations.get(client_uid)
        answering = running is not None and not running.done() and (spec_now is None or spec_now.task is not running)
        ears = heard_task = asd = None
        if (omni_addressee.AUDIO_ENABLED or active_speaker.ENABLED) and audio:
            held = client_uid in self.barge_held
            heard_task = self._start_heard(audio, "pause")
            asd = self._asd_check(audio, time.monotonic(), "pause")
            # The ears are only needed when nobody near faces her
            # (_addressee_policy): ask them now in that case, else not at all.
            _now = time.monotonic()
            need_ears = omni_addressee.AUDIO_ENABLED and face_detect.ATTENTION.presence(
                self.turn_started.get(client_uid, _now), _now
            )[0] is None  # with a camera the policy goes by who stopped in front
            if (answering or held) and self._is_continuation(client_uid):
                # The visitor she was answering went on (_CONT_S): same turn.
                await self._confirm_barge(websocket, client_uid, "the visitor went on")
            elif answering or held:
                # She is answering (or still speaking, the barge-in held), and
                # starting this run would cut her off: see who is talking first.
                ok, why = await self._addressee_policy(client_uid, None, asd, ears_audio=audio)
                if not ok:
                    # (a held barge-in stays held: the endpoint judges the whole utterance)
                    logger.info(f"[spec] skip: not to her ({why}), and she is still answering")
                    return None
                await self._confirm_barge(websocket, client_uid, why)
            elif need_ears:
                ears = asyncio.create_task(omni_addressee.AUDIO.score(audio, "pause"))
        elif (
            (face_detect.ADDRESSEE or self._omni_judged(client_uid))
            and answering
            and self._addressed_now(client_uid)[0] is False
        ):
            logger.info(f"[spec] skip: not to her, and she is still answering ({text[:40]!r})")
            return None
        async with self._spec_lock(client_uid):
            if self.speech_epoch.get(client_uid, 0) != epoch:
                return None
            # Stale if the turn already ended (finalize owns it now) or another
            # segment closed while this ASR ran (a newer _speculate will run).
            if self.segment_asr_tasks.get(client_uid) is None or len(
                self.segment_asr_tasks[client_uid]
            ) != len(tasks):
                logger.info(
                    f"[spec] too late: ASR took {asr_ms:.0f} ms and the endpoint "
                    "(or another segment) came first"
                )
                return None
            current = self.speculations.get(client_uid)
            if current is not None and not placeholder and current.text == text:
                return None
            # Did this turn's speculation cut off a real reply? Asked before an
            # earlier speculation of the same turn is discarded: that one may be
            # what did the cutting, and it carries the answer forward.
            prior = self.current_conversation_tasks.get(client_uid)
            if current is not None and current.task is prior:
                superseded_reply = current.superseded_reply
            else:
                superseded_reply = prior is not None and not prior.done()
            if current is not None:
                self.speculations.pop(client_uid)
                await speculative.discard(current, "transcript grew")
            context = self.client_contexts.get(client_uid)
            if context is None:
                return None
            mem_len, hist_len, mem_tail, mem_ref = speculative.checkpoint(context)
            gate = speculative.GatedSocket(websocket)
            # omni mode: the speculative reply hears the utterance so far (the
            # endpoint's audio only has the tail after this pause added). The
            # HIT test stays on the text.
            spec_audio = audio
            await handle_conversation_trigger(
                msg_type="text-input",
                data={"text": text, **({"audio": spec_audio} if spec_audio else {})},
                client_uid=client_uid,
                context=context,
                websocket=gate,
                client_contexts=self.client_contexts,
                client_connections=self.client_connections,
                chat_group_manager=self.chat_group_manager,
                received_data_buffers=self.received_data_buffers,
                current_conversation_tasks=self.current_conversation_tasks,
                broadcast_to_group=self.broadcast_to_group,
            )
            task = self.current_conversation_tasks.get(client_uid)
            if task is None:
                return None
            spec = speculative.Speculation(
                text=text, task=task, gate=gate, context=context,
                mem_len=mem_len, hist_len=hist_len, mem_tail=mem_tail, mem_ref=mem_ref,
                epoch=epoch, superseded_reply=superseded_reply, placeholder=placeholder,
            )
            spec.ears = ears
            spec.heard = heard_task
            spec.asd = asd
            spec.audio = audio
            if not placeholder:
                spec.text_ready.set()
            self.speculations[client_uid] = spec
            how = f"audio-first, {asr_ms:.0f} ms after the pause" if placeholder else f"{asr_ms:.0f} ms of ASR"
            from .utils.turn_timing import wall_ms

            logger.info(f"[spec] started on the pause ({how}) @{wall_ms()}: {text[:60]!r}")
            return spec

    async def _finalize_and_settle(
        self, websocket: WebSocket, client_uid: str, tasks: list, epoch: int
    ) -> None:
        """_finalize_vad_turn, plus: a speculation it neither confirmed nor
        replaced (the turn was dropped, held and abandoned, ...) is rolled back.
        Only this turn's (`epoch`): the next turn's may already be running."""
        try:
            await self._finalize_vad_turn(websocket, client_uid, tasks, epoch)
        finally:
            await self._drop_speculation(client_uid, "turn ended without it", epoch)

    async def _send_backchannel(self, websocket: WebSocket, client_uid: str) -> None:
        """A short acknowledgment the moment a turn ends, to cover the gap until
        the reply's first audio. Only on the slow path: a speculative HIT
        already has audio ready. BACKCHANNEL=1 to enable (utils/backchannel.py)."""
        context = self.client_contexts.get(client_uid)
        if context is None:
            return
        try:
            from .utils import backchannel  # lazy: optional

            clip = await backchannel.get_clip(context)
            if clip:
                await websocket.send_text(json.dumps({"type": "backchannel", "audio": clip}))
                logger.info("[backchannel] sent")
        except Exception as e:  # noqa: BLE001 — never break a turn over this
            logger.debug(f"[backchannel] skipped: {e!r}")

    async def _finalize_vad_turn(
        self, websocket: WebSocket, client_uid: str, tasks: list, epoch: int = -1
    ) -> None:
        """Join the turn's segment transcripts (in order) and start the
        conversation via the text-input path. Semantic endpointing: a transcript
        that ends mid-thought ("〜が", "〜で", "、" …) is held open for a grace
        period — if the user resumes speaking it folds into the next turn,
        otherwise it fires as-is."""
        if self.early_released.get(client_uid) == epoch:
            # smart-turn already released this turn's reply at the pause; the
            # endpoint only confirms the speaker stayed quiet.
            self.early_released.pop(client_uid, None)
            self.turn_audio.pop(client_uid, None)
            logger.info("[smart-turn] endpoint confirmed an early release")
            return
        turn_timing.start()  # TTFA t0: the endpoint just fired
        # omni mode: this utterance's audio (taken at <|TURN_END|>). Popped here
        # so a dropped turn never lends its audio to the next one. A rapid
        # follow-up merge below joins the TEXTS; the audio is the latest
        # utterance only.
        turn_audio = self.turn_audio.pop(client_uid, None)
        if turn_audio and client_uid in self.render_clients:
            speaker_probe.record(turn_audio)  # log only: whose voice (speaker_probe.py)
            logger.info("[contingency] " + turn_contingency.line(
                self.turn_started.get(client_uid, time.monotonic()), self.turn_end_at.get(client_uid, time.monotonic())))
            _fx = face_detect.ATTENTION.face_x(self.turn_started.get(client_uid, time.monotonic()) - 1.0, time.monotonic())
            logger.info("[direction] " + voice_direction.line(
                self.turn_started.get(client_uid, time.monotonic()), self.turn_end_at.get(client_uid, time.monotonic()))
                + (f" | face x {_fx:.2f}" if _fx is not None else " | no near face"))
        _prev_task = self.current_conversation_tasks.get(client_uid)
        _spec_now = self.speculations.get(client_uid)
        _reply_running = (
            _prev_task is not None and not _prev_task.done()
            and not (_spec_now is not None and _spec_now.task is _prev_task)
        )
        _last_ts, _last_text = self.last_vad_turn.get(client_uid, (0.0, ""))
        _irq = self.interrupted_at.get(client_uid, 0.0)
        _recent_irq = bool(_irq) and _irq > _last_ts and time.monotonic() - _irq < _TURN_MERGE_S
        _would_merge = (_reply_running or _recent_irq) and time.monotonic() - _last_ts < _TURN_MERGE_S and bool(_last_text)
        # A busy room: she answers only what both transcripts heard
        # (_understood_in_busy_room). Never by waiting for Whisper's (+0.5 s,
        # refused 2026-11-13): the fast release goes out as always and is taken
        # back when the transcript arrives and disagrees (_speculate_inner).
        _busy_room = _UNDERSTOOD_GATE and client_uid in self.render_clients and turn_contingency.busy_room()
        if _FAST_RELEASE and turn_audio and not self.pending_vad_transcript.get(client_uid) and not _would_merge:
            spec = self.speculations.get(client_uid)
            # base64 of a 16 kHz int16 WAV: seconds ≈ (bytes - 44) / 32000
            utter_s = (len(turn_audio) * 3 / 4 - 44) / 32000
            heard_now = ""
            if (
                spec is not None
                and spec.epoch == epoch
                and not spec.task.done()
                and not spec.gate.is_open
                and utter_s >= _FAST_RELEASE_MIN_S
            ):
                # The Thinker's words so far end mid-sentence ("So I", "Tell me
                # about"): the visitor paused, not finished. Hold the turn on the
                # path below (it waits _VAD_GRACE_S for them to go on, and joins
                # both halves — text AND audio) instead of answering the half.
                heard_now = await self._spec_heard(spec)
                if await self._unfinished(spec=spec):
                    logger.info(f"[server-vad] fast release held: the words look unfinished ({heard_now[-40:]!r})")
                else:
                    addressed, why = await self._ears_verdict(client_uid, spec, None)
                    if addressed is None and (face_detect.ADDRESSEE or self._omni_judged(client_uid)):
                        addressed, why = await self._addressed_settled(client_uid)
                    if addressed is False:
                        # Not to her by who was talking: nothing is said, nothing
                        # is kept (_finalize_and_settle rolls the run back).
                        logger.info(f"[addressee] not to her ({why}) — dropped at the endpoint")
                        self._drop_barge(client_uid, why)
                        await self._release_dropped_turn(websocket)
                        return
                    _wake_ok, _wake_why = (True, "")
                    if addressed is not False:
                        _wake_ok, _wake_why = self._session_gate(client_uid, [heard_now])
                        if not _wake_ok:
                            # Whisper may still hear the name the Thinker missed:
                            # the slow path below judges with both transcripts.
                            logger.info(f"[wake] fast release held: {_wake_why} (heard {heard_now[-50:]!r})")
                    if addressed is not False and _wake_ok:
                        _bg, _bg_why = await self._bg_talker(client_uid, turn_audio)
                        if _bg_why:
                            logger.info(f"[background] {'DROP' if _bg else 'keep'}: {_bg_why}")
                        if _bg and self._BG_DROP:
                            logger.info(f"[addressee] not to her ({_bg_why}) — dropped at the endpoint")
                            self._drop_barge(client_uid, _bg_why)
                            await self._release_dropped_turn(websocket)
                            return
                        self._bg_checked[client_uid] = epoch
                        why = f"{why}; {_wake_why}" if _wake_why else why
                        logger.info(f"[addressee] {'to her' if addressed else 'unjudged'} ({why})")
                        if _busy_room and spec.text_ready.is_set() and spec.text and spec.text != speculative.SPOKEN_PLACEHOLDER:
                            # Whisper's words for this audio are already in: no need to speak first.
                            _a = turn_contingency.agreement(heard_now, spec.text) if heard_now else 1.0
                            if _a < _UNDERSTOOD_MIN:
                                _uw = (f"busy room and no words both transcripts heard — the two transcripts agree "
                                       f"{_a:.0%}: {heard_now[:50]!r} / {spec.text[:50]!r}")
                                logger.info(f"[addressee] not to her ({_uw}) — dropped at the endpoint")
                                self._drop_barge(client_uid, _uw)
                                await self._release_dropped_turn(websocket)
                                return
                        await self._confirm_barge(websocket, client_uid, why)
                        if await self._release_spec(websocket, client_uid, spec, "at the endpoint", utter_s):
                            turn_timing.mark("asr_done")
                            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parts = _turn_parts(results)
        # Reject whisper's non-speech filler SEGMENT BY SEGMENT, before joining —
        # see _segment_is_hallucination for why the turn-level test below is not
        # enough on its own.
        kept = [p for p in parts if not _segment_is_hallucination(p)]
        if len(kept) != len(parts):
            dropped = [p for p in parts if _segment_is_hallucination(p)]
            logger.info(f"[server-vad] dropped {len(dropped)} hallucinated segment(s): {dropped!r}")
            parts = kept
        held = self.pending_vad_transcript.pop(client_uid, "")
        if held and _OMNI_AUDIO_IN:
            # The visitor went on after a held, unfinished turn: the text joins
            # the halves, and so must the audio — the Thinker answers from the
            # audio, and this utterance's alone is only the second half (the
            # first half reached it only as Whisper's text, which omni does not
            # send). Re-cut from the held utterance's start.
            _prev = self.prev_turn_start.get(client_uid)
            _both = self._audio_b64_from(client_uid, _prev[0]) if _prev is not None else None
            if _both:
                turn_audio = _both
                logger.info("[server-vad] held turn resumed: its audio re-cut from the first half's start")
        if not held and _is_looped_noise(parts):
            logger.info(f"[server-vad] dropped looped-noise turn: {parts!r}")
            await self._release_dropped_turn(websocket)
            return
        _rest = "".join(parts)
        # (English halves need a space between them: "about?The South Market")
        _sep = " " if held and _rest and not held[-1].isspace() and _rest[:1].isascii() and _rest[:1].isalnum() else ""
        text = (held + _sep + _rest).strip()
        if not text:
            logger.debug("[server-vad] turn had no transcribable speech; ignoring")
            await self._release_dropped_turn(websocket)
            return
        _t = text.strip()
        if len(_t) <= 40 and any(h in _t.lower() for h in _ASR_HALLUCINATIONS):
            logger.info(f"[server-vad] dropped ASR hallucination turn: {_t!r}")
            await self._release_dropped_turn(websocket)
            return
        if _is_junk_transcript(_t):
            logger.info(f"[server-vad] dropped junk/noise turn: {_t!r}")
            await self._release_dropped_turn(websocket)
            return
        # WRONG-SCRIPT GUARD: the local SenseVoice model is multilingual and its
        # language pin is a NO-OP in this build (verified: en/ja/zh all decode
        # identically), so on quiet or distant audio it happily emits Chinese
        # word-salad — "很关之。", "我开没是这。" — which the avatar then
        # answered. For an English character any CJK-heavy transcript is by
        # definition a mis-decode, so drop it instead of replying to it.
        _lang = getattr(
            self.client_contexts.get(client_uid).character_config, "language", "ja"
        ) if self.client_contexts.get(client_uid) else "ja"
        if _lang != "ja" and _CJK_CHARS_RE.search(_t):
            cjk = len(_CJK_CHARS_RE.findall(_t))
            if cjk >= max(2, len(_t.replace(" ", "")) * 0.3):
                logger.info(
                    f"[server-vad] dropped wrong-script turn ({cjk} CJK chars in "
                    f"an {_lang} session — ASR mis-decode): {_t!r}"
                )
                await self._release_dropped_turn(websocket)
                return
        # ADAPTIVE ENDPOINT: the hard VAD endpoint is short (~0.45s) now; how
        # much longer we hold depends on how FINISHED the transcript looks.
        hold_s = 0.0
        # The Thinker's transcript counts too: Whisper turns a broken-off "So I"
        # into "I'm sorry." (11-06 13:31:36), which looks finished.
        _spec_e = self.speculations.get(client_uid)
        _spec_e = _spec_e if _spec_e is not None and _spec_e.epoch == epoch and not held else None
        # (After a held first half, only the new part: the first half was already
        # found unfinished, and with it in front the check called "Can you tell me
        # about? The South Market, please." unfinished too and held it again.)
        _unf_text = _rest.strip() if held else text
        _votes: list = []
        _any_unf = await self._unfinished(_unf_text, spec=_spec_e, votes=_votes)
        if _any_unf and all(_votes) and not held:
            hold_s = max(_VAD_GRACE_S, _UNFINISHED_HOLD_S)  # clearly mid-thought (〜が, 〜で, "about", trailing 、)
        elif _any_unf and held:
            # The second half of a held turn: they have paused once already, and
            # a fragment alone ("Prospective.") reads unfinished — 1.5 s more made
            # that reply 2.1 s late (2026-11-09 03:24). The usual grace only.
            hold_s = _VAD_GRACE_S
        elif _any_unf:
            # The two transcripts disagree (2026-11-09 02:54: the Thinker heard
            # "Ah, which ah, which job can you ah, can you explain more?" and
            # called the stammer unfinished, Whisper's "Which of them can you
            # explain more?" was finished): the usual grace, not the long hold —
            # she answered a finished question 1.8 s late.
            hold_s = _VAD_GRACE_S
        elif not _STRONG_END_RE.search(text.strip()):
            hold_s = _AMBIG_GRACE_S  # no clear sentence end — wait a bit more
        # Whose voice it was is settled BEFORE a turn is held for its second
        # half. The hold used to come first: a broken-off phrase from the room
        # ("Zakir na.", 2026-11-09 22:48:23, nobody in front) was kept as an
        # unfinished turn of "the visitor", so the next sound counted as them
        # going on — she was told to stop at its first syllable, whoever she
        # was answering — and its words would have been joined onto the next
        # real question. Only a turn that is hers to take is held.
        _pre: tuple | None = None
        if hold_s > 0 and not held and not _would_merge:
            _pa, _pw = await self._ears_verdict(client_uid, _spec_e, turn_audio)
            if _pa is None and (face_detect.ADDRESSEE or self._omni_judged(client_uid)):
                _pa, _pw = await self._addressed_settled(client_uid)
            if _pa is not False and self._bg_checked.get(client_uid) != epoch:
                _bg, _bg_why = await self._bg_talker(client_uid, turn_audio)
                if _bg_why:
                    logger.info(f"[background] {'DROP' if _bg else 'keep'}: {_bg_why}")
                if _bg and self._BG_DROP:
                    _pa, _pw = False, _bg_why
                else:
                    self._bg_checked[client_uid] = epoch
            if _pa is False:
                logger.info(f"[addressee] not to her ({_pw}) — ignored, not held: {text!r}")
                self._drop_barge(client_uid, _pw)
                await self._release_dropped_turn(websocket)
                return
            _pre = (_pa, _pw)
        if hold_s > 0 and self.speech_epoch.get(client_uid, 0) != epoch and self.in_utterance.get(client_uid):
            # They went on before the hold below could start (the transcript came
            # 0.5 s after the endpoint, the pause was 0.9 s): nothing would cancel
            # the hold, and she answered the half — "Can you tell me about?" —
            # before the merge caught up (11-07 probe). Hand it to the utterance
            # under way, as a cancelled hold does.
            self.pending_vad_transcript[client_uid] = text
            logger.info(f"[server-vad] unfinished turn joins the utterance under way: {text[-40:]!r}")
            return
        if hold_s > 0:
            self.pending_vad_transcript[client_uid] = text
            grace = asyncio.create_task(self._hold_wait(client_uid, hold_s))
            self.vad_grace_tasks[client_uid] = grace
            logger.debug(
                f"[server-vad] holding turn {hold_s}s (ending: {text[-12:]!r})"
            )
            try:
                await grace
            except asyncio.CancelledError:
                return  # user resumed; held text joins the next turn
            finally:
                self.vad_grace_tasks.pop(client_uid, None)
            text = self.pending_vad_transcript.pop(client_uid, text)
        # Speech among the visitors rather than to her: drop it like noise. The
        # speculation already running for it is rolled back by
        # _finalize_and_settle; one that IS addressed is untouched.
        # A turn that continues the previous one (_would_merge) is judged on
        # BOTH utterances: heard alone its tail is "a fragment" (0/9) even when
        # it finishes a question to her, and skipping the check let room
        # chatter that had once been answered chain on and on (2026-11-04
        # 17:37: four merged chats in a row, each answered). Right after a turn
        # she took (_CONT_S) it is the same visitor going on: only seeing that
        # nobody in front is speaking drops it.
        _spec_t = self.speculations.get(client_uid)
        if _would_merge:
            _prev = self.prev_turn_start.get(client_uid)
            _both = self._audio_b64_from(client_uid, _prev[0]) if _prev is not None else None
            if self._is_continuation(client_uid):
                _asd = self._asd_check(_both or turn_audio, self.turn_end_at.get(client_uid, time.monotonic()), "continuation")
                addressed, why = await self._addressee_policy(client_uid, None, _asd)
                why = f"{why}; continues the turn she took"
            else:
                addressed, why = await self._ears_verdict(client_uid, None, _both or turn_audio)
                why = f"{why} on the merged utterances"
        elif _pre is not None:
            addressed, why = _pre  # judged before the hold, on this same utterance
        else:
            addressed, why = await self._ears_verdict(
                client_uid, _spec_t if _spec_t is not None and _spec_t.epoch == epoch and not held else None,
                turn_audio,
            )
        if addressed is None and (face_detect.ADDRESSEE or self._omni_judged(client_uid)):
            addressed, why = await self._addressed_settled(client_uid)
        if addressed is not False:
            _heard_t = ""
            if _spec_t is not None and _spec_t.heard is not None and _spec_t.heard.done():
                try:
                    _heard_t = (_spec_t.heard.result() or (None, ""))[1] or ""
                except Exception:  # noqa: BLE001
                    _heard_t = ""
            _wake_ok, _wake_why = self._session_gate(client_uid, [text, held, _heard_t])
            if not _wake_ok:
                logger.info(f"[addressee] not to her ({why}; {_wake_why}) — ignored: {text!r}")
                self._drop_barge(client_uid, _wake_why)
                await self._release_dropped_turn(websocket)
                if addressed and face_detect.ATTENTION.engaged(
                    self.turn_started.get(client_uid, time.monotonic()), time.monotonic()
                )[0]:
                    await self._maybe_invite(websocket, client_uid)
                return
            if _wake_why:
                why = f"{why}; {_wake_why}"
            if self._bg_checked.get(client_uid) != epoch:
                _bg, _bg_why = await self._bg_talker(client_uid, turn_audio)
                if _bg_why:
                    logger.info(f"[background] {'DROP' if _bg else 'keep'}: {_bg_why}")
                if _bg and self._BG_DROP:
                    addressed, why = False, _bg_why
        if addressed is not False and _busy_room and turn_audio:
            _ok, _uw = await self._understood_in_busy_room(turn_audio, text)
            if not _ok:
                addressed, why = False, _uw
            else:
                why = f"{why}; {_uw}"
        if addressed is False:
            logger.info(f"[addressee] not to her ({why}) — ignored: {text!r}")
            self._drop_barge(client_uid, why)
            await self._release_dropped_turn(websocket)
            return
        await self._confirm_barge(websocket, client_uid, why)
        self._mark_accepted(client_uid)
        if addressed is not None or why != "ears off":
            logger.info(f"[addressee] {'to her' if addressed else 'unjudged'} ({why})")
        turn_timing.mark("asr_done")
        # (The 相槌 is sent just before the reply starts — _send_backchannel,
        # opt-in via BACKCHANNEL=1; it was removed once by request, then asked for again.)
        # Rapid follow-up: the user kept talking after a pause long enough for
        # the VAD to close the turn, so a SECOND (third...) turn fired while the
        # reply to the first was still running. Merge the texts and answer ONCE
        # (handle_conversation_trigger cancels the stale running reply).
        # The running task may be THIS turn's own speculation, which says
        # nothing about the previous reply: ask the speculation whether it
        # superseded one. Read as "the previous reply is still running", it
        # merged a turn into one that had finished or been interrupted, and the
        # LLM was asked the old question again (2026-10-23 20:22:04).
        prev_task = self.current_conversation_tasks.get(client_uid)
        own_spec = self.speculations.get(client_uid)
        if own_spec is not None and own_spec.epoch == epoch and own_spec.task is prev_task:
            reply_running = own_spec.superseded_reply
        else:
            reply_running = prev_task is not None and not prev_task.done()
        last_ts, last_text = self.last_vad_turn.get(client_uid, (0.0, ""))
        _irq = self.interrupted_at.get(client_uid, 0.0)
        recently_interrupted = bool(_irq) and _irq > last_ts and time.monotonic() - _irq < _TURN_MERGE_S
        if (
            (reply_running or recently_interrupted)
            and time.monotonic() - last_ts < _TURN_MERGE_S
            and last_text
        ):
            text = f"{last_text} {text}".strip()
            self.interrupted_at.pop(client_uid, None)
            logger.info(
                f"[server-vad] rapid follow-up — merged turn{' (after a barge-in)' if recently_interrupted else ''}: {text!r}"
            )
            # omni: the audio must cover BOTH utterances, else the Thinker hears
            # only this one while the words say both.
            _prev = self.prev_turn_start.get(client_uid)
            if _OMNI_AUDIO_IN and _prev is not None and time.monotonic() - _prev[1] < _TURN_MERGE_S + 30:
                _merged = self._audio_b64_from(client_uid, _prev[0])
                if _merged:
                    turn_audio = _merged
                    logger.info("[server-vad] merged turn audio re-cut from the previous utterance's start")
        self.last_vad_turn[client_uid] = (time.monotonic(), text)
        logger.info(f"[server-vad] turn transcript: {text!r}")
        asr_capture.note(f"TURN {text}")
        # Tell the client what it just said, so the chat panel shows the user's turn.
        # It cannot know: server VAD transcribes on this side and then enters the
        # normal pipeline as a plain "text-input", which is indistinguishable from
        # something the user TYPED — and typed text is appended locally by
        # use-text-input, so conversation_utils only announces a transcription when
        # it does the ASR itself (np.ndarray input). Neither branch fires for a spoken
        # turn, which is why speech never appeared in the history while tool calls did.
        # Sent from here rather than widened downstream, so typing stays single-entry.
        try:
            await websocket.send_text(
                json.dumps({"type": "user-input-transcription", "text": text})
            )
        except Exception as e:  # noqa: BLE001 — never let the panel break the turn
            logger.warning(f"[server-vad] could not send transcription: {e}")
        # An audio-first speculation stored a placeholder; its transcript is this
        # same ASR, so the swap is moments away — wait for it so the HIT test
        # below compares words, not the placeholder.
        pending = self.speculations.get(client_uid)
        text_late = False
        if pending is not None and pending.placeholder:
            try:
                await asyncio.wait_for(pending.text_ready.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                # The run is still the audio of this very utterance; comparing
                # words would only fail on the placeholder and throw it away
                # for a cold start (1.7-2.4 s). Take it.
                text_late = True
                logger.warning("[spec] transcript never caught up; taking the run on the audio alone")
        async with self._spec_lock(client_uid):
            spec = self.speculations.get(client_uid)
            if spec is not None and spec.epoch != epoch:
                spec = None  # the next turn's — not ours to take or drop
            else:
                self.speculations.pop(client_uid, None)
            if (
                spec is not None
                and (spec.text.strip() == text.strip() or (text_late and spec.placeholder))
                and not spec.task.cancelled()
            ):
                # The reply has been running since the pause; publish it.
                head_ms = (time.monotonic() - spec.started) * 1000
                held = spec.gate.held
                # A HIT whose voice is not synthesized yet (GSV needs 0.4-0.9 s
                # for its first chunk; the head start is ~0.4 s) still leaves a
                # silent gap after the endpoint — cover it with the 相槌, as the
                # slow path does. Qwen3-TTS (0.1 s) rarely gets here.
                if not spec.gate.has_audio:
                    await self._send_backchannel(websocket, client_uid)
                await spec.gate.release()
                self._watch_reply(client_uid, spec.task)
                logger.info(
                    f"[spec] HIT: reply started {head_ms:.0f} ms before the endpoint, "
                    f"{held} msgs released at once"
                )
                return
            if spec is not None:
                # Roll it back BEFORE the real turn starts writing history.
                await speculative.discard(spec, "endpoint on a different transcript")
            await self._send_backchannel(websocket, client_uid)
            await handle_conversation_trigger(
                msg_type="text-input",
                data={"text": text, **({"audio": turn_audio} if turn_audio else {})},
                client_uid=client_uid,
                context=self.client_contexts[client_uid],
                websocket=websocket,
                client_contexts=self.client_contexts,
                client_connections=self.client_connections,
                chat_group_manager=self.chat_group_manager,
                received_data_buffers=self.received_data_buffers,
                current_conversation_tasks=self.current_conversation_tasks,
                broadcast_to_group=self.broadcast_to_group,
            )
            self._watch_reply(client_uid, self.current_conversation_tasks.get(client_uid))

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)
            # The switch rebuilt the agent (fresh, empty memory) — re-attach the
            # NEW character's own most recent conversation so each of them keeps
            # her own continuous thread across switches.
            self._resume_latest_history(context)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        request_client_uid = data.get("client_uid") or data.get("client_id") or client_uid
        context = self.client_contexts.get(request_client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": request_client_uid,
                }
            )
        )

    async def _handle_audio_caps(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Record what this client can decode, so the WS audio path can compress.

        Opus decoding in a browser needs WebCodecs, which is not universal, so
        the client declares it rather than the server assuming it. Saying nothing
        means raw PCM — the behaviour every existing client already has.
        """
        supported = bool(data.get("opus"))
        ws_opus.set_client_opus(websocket, supported)
        logger.info(
            f"[ws-opus] client {client_uid} "
            f"{'can' if supported else 'cannot'} decode Opus"
            + (f" ({ws_opus.opus_bitrate()} bps)" if supported else "")
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")
