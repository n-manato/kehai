from typing import Union, List, Dict, Any, Optional
import asyncio
import json
import random
import re
from loguru import logger
import numpy as np

from . import tool_notify
from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    safe_websocket_send,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext

# Import necessary types from agent outputs
from ..agent.output_types import DisplayText, SentenceOutput, AudioOutput

# Spoken the moment a TOOL starts executing: the decide->execute->re-prompt loop
# is otherwise dead air for 6-30s ("ツールを使うとしゃべらなくなる"). Rotating a
# few phrasings keeps it from sounding canned. Picked by character language.
_TOOL_FILLERS = {
    "ja": [
        "ちょっと調べてみるね!",
        "待っててね、いま確認してみる!",
        "うーん、ちょっと見てみるね!",
    ],
    "en": [
        "One moment. Retrieving that information.",
        "Let me check. This will only take a few seconds.",
        "Accessing the relevant records now.",
    ],
}


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Make sure the shared GSV server carries THIS character's fine-tune before any
    # sentence goes out. gsv_stream_config() below only carries the reference clip and
    # languages; the weights are server-side state, and leaving that swap to a
    # fire-and-forget task meant a switched-in character could speak the whole first
    # reply through the previous character's model. Deduped + bounded — free after the
    # first turn, and it will not stall a reply behind a cold gateway.
    await context.ensure_gsv_weights()

    # Create TTSTaskManager for this conversation (with the character's
    # streaming voice profile — ref clip / languages / weights).
    tts_manager = TTSTaskManager(stream_cfg=context.gsv_stream_config())
    full_response = ""  # Initialize full_response here

    # Tool-start filler: speaks through the SAME per-sentence pipeline (ordered
    # send chain, WebRTC track), so it behaves like any other sentence of the
    # reply. Registered per turn; cleared in finally.
    async def _speak_tool_filler(tool_name: str) -> None:
        try:
            lang = getattr(context.character_config, "language", "ja")
            txt = random.choice(_TOOL_FILLERS.get(lang, _TOOL_FILLERS["ja"]))
            logger.info(f"[tool-filler] speaking {txt!r} (tool: {tool_name})")
            await tts_manager.speak(
                tts_text=txt,
                display_text=DisplayText(
                    text=txt, name=context.character_config.character_name
                ),
                actions=None,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
            )
        except Exception as e:  # noqa: BLE001 — filler must never break the turn
            logger.warning(f"[tool-filler] failed: {e!r}")

    tool_notify.set_callback(_speak_tool_filler)

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await safe_websocket_send(
                websocket_send,
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                ),
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await safe_websocket_send(
                websocket_send, json.dumps({"type": "backend-synth-complete"})
            )
            # Phase G: the WebRTC track drains server-side — tell the frontend
            # exactly when this turn's audio has finished playing.
            from ..realtime import webrtc_manager
            from ..realtime.webrtc_audio import notify_when_drained

            _track = webrtc_manager.track_for_send(websocket_send)
            if _track is not None:
                asyncio.create_task(notify_when_drained(websocket_send, _track))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        # Never persist a CONTENTLESS reply. An interrupted turn leaves "..."
        # (or an empty string), and once a few of those sit in the history the
        # model reads them as a legitimate answer shape and starts emitting bare
        # "..." itself — the "たまに帰ってこない" turns. Keep them out of the
        # record so the pattern cannot feed itself.
        _speech = re.sub(r"[.…\s]", "", full_response or "")
        # An "announce-only" turn — a short line that ends on a dash/colon and
        # never delivers the substance ("Taking that as a request for Japanese
        # news —"). Once a few of those are in the history the model copies the
        # shape and stops mid-thought on every turn, so they are not recorded.
        _stub = bool(
            full_response
            and (
                # ends on a dash/colon and never delivers
                re.search(r"[—–:：]\s*$", full_response.strip())
                # or PROMISES a lookup and carries no data: short, mentions
                # retrieving/searching, and contains no digits. Leaving these in
                # the transcript taught the model to answer "I'm retrieving the
                # latest news" INSTEAD of calling the search tool.
                or (
                    re.search(
                        r"\b(retriev\w*|search\w*|look\w* up|fetch\w*|check\w*)\b",
                        full_response,
                        re.I,
                    )
                    and not re.search(r"\d", full_response)
                )
            )
            and len(full_response.split()) < 20
        )
        if context.history_uid and full_response and (not _speech or _stub):
            logger.info(
                f"[history] not storing {'announce-only' if _stub else 'contentless'} "
                f"reply {full_response[:60]!r}"
            )
        elif context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.exception(f"Error in conversation chain: {e}")
        await safe_websocket_send(
            websocket_send,
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"}),
        )
        raise
    finally:
        tool_notify.clear()  # stale filler must never speak into a later turn
        cleanup_conversation(tts_manager, session_emoji)
