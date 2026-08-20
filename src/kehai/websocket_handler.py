from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import os
import re
import time
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
from .conversations.conversation_handler import (
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
            vad_engine=self.default_context_cache.vad_engine,
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
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        context = self.client_contexts.get(client_uid)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

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
        await self._process_vad_chunk(websocket, client_uid, chunk)

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
            f"origin={data.get('origin', '?')}{note}"
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
        for audio_bytes in context.vad_engine.detect_speech(chunk):
            if audio_bytes == b"<|PAUSE|>":
                # New turn began (IDLE->ACTIVE). Drop any tasks left over from
                # an aborted turn (e.g. mic toggled off mid-utterance) so they
                # can't leak into this turn's transcript. Barge-in interruption
                # stays client-driven (its own VAD already interrupts).
                self.segment_asr_tasks.pop(client_uid, None)
                # Free PC-B GPU1 for THIS turn's ASR + TTS: the previous reply's
                # in-flight motion generations are obsolete the moment the user
                # starts talking, but were measured still draining (queued
                # +4.3s) — slowing whisper (asr 250 -> 1546ms) and GSV's first
                # chunk (500 -> ~700ms). Cancelling closes their gateway
                # streams, which aborts the service-side queue within ~0.2s.
                try:
                    from . import motion_cache  # lazy: optional subsystem

                    motion_cache.cancel_all()
                except Exception:  # noqa: BLE001 — never break the VAD path
                    pass
                # Semantic endpointing: the user resumed while an incomplete
                # transcript was in its grace window — cancel the pending fire;
                # the held text will prepend to this new turn's transcript.
                grace = self.vad_grace_tasks.pop(client_uid, None)
                if grace is not None and not grace.done():
                    grace.cancel()
            elif audio_bytes == b"<|RESUME|>":
                pass
            elif audio_bytes.startswith(b"<|SEG|>"):
                if context.asr_engine is None:
                    continue
                pcm = np.frombuffer(audio_bytes[len(b"<|SEG|>") :], dtype=np.int16)
                segment = pcm.astype(np.float32) / 32768.0
                logger.debug(
                    f"[server-vad] segment closed ({len(segment) / 16000:.2f}s) "
                    "-> ASR while user keeps talking"
                )
                self.segment_asr_tasks.setdefault(client_uid, []).append(
                    asyncio.create_task(context.asr_engine.async_transcribe_np(segment))
                )
            elif audio_bytes == b"<|TURN_END|>":
                tasks = self.segment_asr_tasks.pop(client_uid, [])
                if tasks:
                    asyncio.create_task(
                        self._finalize_vad_turn(websocket, client_uid, tasks)
                    )

    def _resume_latest_history(self, context: ServiceContext) -> None:
        """Attach the newest existing conversation for this character, or start
        a fresh one. Called on every connection so the thread survives reloads,
        restarts and character switches back and forth.

        Long threads are safe: the agent's own compaction (LLM_COMPACT_AT)
        summarises old turns in the background, and the per-request context trim
        is the final backstop before the model call."""
        try:
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
        transcribes as garbage), which is why it showed up there first."""
        try:
            await websocket.send_text(
                json.dumps({"type": "control", "text": "conversation-chain-end"})
            )
        except Exception as e:  # noqa: BLE001 — client may already be gone
            logger.debug(f"[server-vad] could not release dropped turn: {e!r}")

    async def _finalize_vad_turn(
        self, websocket: WebSocket, client_uid: str, tasks: list
    ) -> None:
        """Join the turn's segment transcripts (in order) and start the
        conversation via the text-input path. Semantic endpointing: a transcript
        that ends mid-thought ("〜が", "〜で", "、" …) is held open for a grace
        period — if the user resumes speaking it folds into the next turn,
        otherwise it fires as-is."""
        turn_timing.start()  # TTFA t0: the endpoint just fired
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parts = [r.strip() for r in results if isinstance(r, str) and r.strip()]
        # Reject whisper's non-speech filler SEGMENT BY SEGMENT, before joining —
        # see _segment_is_hallucination for why the turn-level test below is not
        # enough on its own.
        kept = [p for p in parts if not _segment_is_hallucination(p)]
        if len(kept) != len(parts):
            dropped = [p for p in parts if _segment_is_hallucination(p)]
            logger.info(f"[server-vad] dropped {len(dropped)} hallucinated segment(s): {dropped!r}")
            parts = kept
        held = self.pending_vad_transcript.pop(client_uid, "")
        if not held and _is_looped_noise(parts):
            logger.info(f"[server-vad] dropped looped-noise turn: {parts!r}")
            await self._release_dropped_turn(websocket)
            return
        text = held + "".join(parts)
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
        if _transcript_looks_incomplete(text):
            hold_s = _VAD_GRACE_S  # clearly mid-thought (〜が, 〜で, trailing 、)
        elif not _STRONG_END_RE.search(text.strip()):
            hold_s = _AMBIG_GRACE_S  # no clear sentence end — wait a bit more
        if hold_s > 0:
            self.pending_vad_transcript[client_uid] = text
            grace = asyncio.create_task(asyncio.sleep(hold_s))
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
        turn_timing.mark("asr_done")
        # (An instant pre-synthesized 相槌 was tried here and REMOVED by user
        # request — the extra utterances weren't wanted. utils/backchannel.py
        # remains available behind BACKCHANNEL=1 wiring if ever revisited.)
        # Rapid follow-up: the user kept talking after a pause long enough for
        # the VAD to close the turn, so a SECOND (third...) turn fired while the
        # reply to the first was still running. Merge the texts and answer ONCE
        # (handle_conversation_trigger cancels the stale running reply).
        prev_task = self.current_conversation_tasks.get(client_uid)
        last_ts, last_text = self.last_vad_turn.get(client_uid, (0.0, ""))
        if (
            prev_task is not None
            and not prev_task.done()
            and time.monotonic() - last_ts < _TURN_MERGE_S
            and last_text
        ):
            text = f"{last_text} {text}".strip()
            logger.info(f"[server-vad] rapid follow-up — merged turn: {text!r}")
        self.last_vad_turn[client_uid] = (time.monotonic(), text)
        logger.info(f"[server-vad] turn transcript: {text!r}")
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
        await handle_conversation_trigger(
            msg_type="text-input",
            data={"text": text},
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
