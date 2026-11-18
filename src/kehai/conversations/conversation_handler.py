import asyncio
import json
from typing import Dict, Optional, Callable

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import modify_latest_message, store_message
from ..utils.sentence_spacing import unglue
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader


# Per-language overrides so the avatar speaks whatever language the UI is set to.
# The LLM directive is appended to the (cached) base system prompt — memory is
# preserved. The TTS language switches GPT-SoVITS synthesis. Japanese needs no
# directive because the persona itself is written in Japanese.
_LLM_LANG_DIRECTIVE = {
    "en": (
        "\n\n## Response language (override)\n"
        "You MUST respond ONLY in natural, fluent English. Ignore any instruction "
        "above to write in Japanese — keep the SAME personality, warmth and "
        "politeness, just express it in English. Do not mix in Japanese."
    ),
    "ja": "",
    "zh": (
        "\n\n## Response language (override)\n"
        "You MUST respond ONLY in natural Simplified Chinese, keeping the same "
        "personality and tone. Do not mix in other languages."
    ),
}
_TTS_LANG = {"en": "en", "ja": "ja", "zh": "zh"}


def apply_client_language(context: ServiceContext, language: Optional[str]) -> None:
    """
    Make the avatar's spoken language follow the UI-selected language by:
    - re-pointing the LLM system prompt at that language (memory kept), and
    - switching the TTS synthesis language.
    No-op when language is unset. Safe if the agent/TTS don't support it.
    """
    if not language:
        logger.info(
            "apply_client_language: no language in message (defaulting to persona)"
        )
        return
    lang = str(language).split("-")[0].lower()  # "en-US" -> "en"

    agent = getattr(context, "agent_engine", None)
    llm_applied = False
    if agent is not None and hasattr(agent, "set_system"):
        # Capture the BASE system prompt (persona + tools) once, before we ever
        # append a language directive. context.system_prompt is None for contexts
        # cloned from cache, so fall back to the agent's current _system (which is
        # the base on first call). Cache it on the agent so re-applying never
        # stacks directives.
        base_prompt = getattr(agent, "_lang_base_system", None)
        if base_prompt is None:
            base_prompt = (
                getattr(context, "system_prompt", None)
                or getattr(agent, "_system", "")
                or ""
            )
            agent._lang_base_system = base_prompt
        try:
            agent.set_system(base_prompt + _LLM_LANG_DIRECTIVE.get(lang, ""))
            llm_applied = bool(base_prompt)
        except Exception as e:
            logger.warning(f"Could not set agent response language to '{lang}': {e}")

    tts = getattr(context, "tts_engine", None)
    tts_lang = _TTS_LANG.get(lang)
    tts_applied = False
    if tts is not None and tts_lang and hasattr(tts, "text_lang"):
        tts.text_lang = tts_lang
        tts_applied = True

    logger.info(
        f"apply_client_language: language='{language}' -> lang='{lang}' | "
        f"LLM set_system={llm_applied} (agent={type(agent).__name__ if agent else None}) | "
        f"TTS text_lang={'=' + tts_lang if tts_applied else 'NOT applied'}"
    )


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")
    # omni mode: the server VAD attaches the utterance audio (base64 WAV) so the
    # speech-in LLM hears the turn itself; absent for typed text and in cascade.
    audio = data.get("audio")
    session_emoji = np.random.choice(EMOJI_LIST)

    # Spoken language: the CHARACTER's configured language wins (the persona,
    # ASR hint, TTS profile and prompts are all keyed to it); the client's UI
    # language is only a fallback for characters that don't declare one.
    apply_client_language(
        context,
        getattr(context.character_config, "language", None) or data.get("language"),
    )

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    session_emoji=session_emoji,
                    metadata=metadata,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        prev = current_conversation_tasks.get(client_uid)
        if prev is not None and not prev.done():
            # The user spoke again before the previous reply finished: the
            # NEWEST input wins. Without this the old task kept running (only
            # its dict reference was overwritten), so 2-3 replies streamed TTS
            # into the same WebRTC track SIMULTANEOUSLY — heard as garbled,
            # interleaved speech ("同時に3つの発話に3つのプロンプト").
            prev.cancel()
            logger.info("[conversation] superseded a running reply with the new turn")
            try:
                from ..realtime import webrtc_manager  # lazy: avoid import cycle

                webrtc_manager.clear_audio(client_uid)  # stop the stale voice NOW
            except Exception:  # noqa: BLE001
                pass
            try:
                from .. import motion_cache  # lazy

                motion_cache.cancel_all()  # stale gestures die with the reply
            except Exception:  # noqa: BLE001
                pass
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
                audio=audio,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        running = task is not None and not task.done()
        heard_response = unglue((heard_response or "").strip())
        if not running and not heard_response:
            # Nothing generating and nothing being heard: no reply to cut off.
            # Stored anyway, a stray barge-in (a state left at "thinking") put a
            # marker in the file each time — 5 in a row on 2026-10-23 22:38-22:39.
            logger.debug("[interrupt] no reply running or playing — ignored")
            return
        if running:
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid and not running:
            # The reply had finished generating and was stored whole; she was
            # cut off while it was still PLAYING. Keep only what she got out,
            # as the agent's memory now does — appended, it stood in the file
            # twice (full, then partial).
            if not modify_latest_message(
                context.character_config.conf_uid, context.history_uid, "ai", heard_response
            ):
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=heard_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )
        elif context.history_uid:
            # Only record what she ACTUALLY got out before being cut off. A
            # barge-in during the pause before the first word leaves
            # heard_response empty, and storing that put a contentless `ai` turn
            # in the file — 165 of them accumulated in one session, every one of
            # them re-read and skipped on load ("Skipping invalid message from
            # history"), and none of them reaching the model.
            #
            # That is how the transcript came to be user-only: measured 45
            # consecutive human turns with no assistant turn between them. The
            # model, shown a conversation in which it had never spoken, started
            # replying "..." — which is contentless, so it was not stored either,
            # which made the next context worse. The loop did not recover.
            if heard_response.strip():
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=heard_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
