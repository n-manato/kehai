import os
import re
from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import get_history
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor


def _filter_tool_call_text(
    chunk: str, buf: str, in_tool: bool
) -> tuple[str, str, bool]:
    """Streaming suppressor for RAW '<tool_call>...</tool_call>' emitted as plain
    text (the hermes template leaks it when tools are withheld). Returns
    (text_to_emit, carry_buffer, in_tool). Tags may split across chunks, so a
    small tail that could be a tag prefix is held back in `buf` until resolved."""
    OPEN, CLOSE = "<tool_call>", "</tool_call>"
    s = buf + chunk
    out = []
    while s:
        if in_tool:
            end = s.find(CLOSE)
            if end == -1:
                # keep only a possible partial CLOSE at the tail; drop the rest
                keep = 0
                for k in range(min(len(CLOSE) - 1, len(s)), 0, -1):
                    if CLOSE.startswith(s[-k:]):
                        keep = k
                        break
                return "".join(out), s[-keep:] if keep else "", True
            s = s[end + len(CLOSE):]
            in_tool = False
        else:
            start = s.find(OPEN)
            if start == -1:
                # emit all but a possible partial OPEN at the tail
                keep = 0
                for k in range(min(len(OPEN) - 1, len(s)), 0, -1):
                    if OPEN.startswith(s[-k:]):
                        keep = k
                        break
                if keep:
                    out.append(s[:-keep])
                    return "".join(out), s[-keep:], False
                out.append(s)
                return "".join(out), "", False
            out.append(s[:start])
            s = s[start + len(OPEN):]
            in_tool = True
    return "".join(out), "", in_tool


# Injected when the tool-round budget is spent, so the model answers from the
# results it already has instead of emitting further (now-disabled) tool calls
# as plain text. Keyed by the character's language — the JA notice explicitly
# forced a JAPANESE reply, which is wrong for an English character.
_TOOL_CAP_NOTICE = {
    "ja": (
        "ツールの使用回数が上限に達しました。これ以上ツールを呼ばず、これまでに"
        "得た結果だけを使って、今すぐ日本語で音声向けの返答をしてください。"
        "<tool_call> やJSONを書いてはいけません。"
    ),
    "en": (
        "The tool-call budget is spent. Do not call any more tools. Answer NOW "
        "in natural spoken English using only the results you already have. "
        "Never write <tool_call> or JSON."
    ),
}


# How much of an unanswered run of user turns to carry forward. Unbounded
# concatenation was a real failure: interrupted turns leave the user message
# with no assistant reply, so run after run merged into ONE message until the
# model was handed ten repetitions of "Japanese news." and answered with a stub.
# Only the newest lines matter — the speaker has moved on from the rest.
_MERGE_KEEP_LINES = 2
_MERGE_MAX_CHARS = 400


def _merge_user_turns(previous: str, new: str) -> str:
    """Fold an unanswered user turn into the next one, newest-last and bounded."""
    lines = [ln for ln in f"{previous}\n{new}".split("\n") if ln.strip()]
    lines = lines[-_MERGE_KEEP_LINES:]
    merged = "\n".join(lines)
    if len(merged) > _MERGE_MAX_CHARS:
        merged = merged[-_MERGE_MAX_CHARS:]
    return merged


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._memory = []
        self._compacting = False  # background-summary in flight (dedupe guard)
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        # Character language ("ja"/"en") — mirrors character_config.language via
        # tts_preprocessor_config (propagated in service_context.load_from_config).
        # Drives the tool-cap notice and the background-summary prompt language.
        self._language = getattr(tts_preprocessor_config, "language", "ja") or "ja"
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        # A CONTENTLESS assistant turn — "" or bare punctuation like "..." — is
        # already refused by the disk rule (single_conversation drops it before
        # store_message), but it used to stay in _memory anyway, and _memory is
        # what gets replayed to the LLM. A handful of `user -> assistant:"..."`
        # pairs in context is enough for qwen3 to read "..." as the answer SHAPE
        # and emit it unprompted, with no interrupt involved; that reply is then
        # dropped from disk again, so the transcript grows more one-sided every
        # turn and she stops speaking altogether. Measured on 2026-08-13: 138
        # contentless replies in a day and 159 consecutive turns with no stored
        # reply. Keeping the RAM rule identical to the disk rule is what makes
        # the loop impossible — no single "..." can ever become a few-shot example.
        if role == "assistant" and not re.sub(r"[.…\s]", "", text_content or ""):
            return

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        # MERGE consecutive USER turns. A turn whose reply was interrupted or
        # superseded stores the human message but never an assistant one, so
        # unanswered questions piled up and the model started answering an OLD
        # one — replies visibly lagged a turn behind. Folding them into a single
        # user turn keeps the context but makes the LATEST request the one being
        # answered. Assistant messages are left alone (streaming appends one per
        # turn by design).
        if (
            role == "user"
            and self._memory
            and self._memory[-1]["role"] == "user"
            and isinstance(self._memory[-1].get("content"), str)
            and isinstance(text_content, str)
        ):
            self._memory[-1]["content"] = _merge_user_turns(
                self._memory[-1]["content"], text_content
            )
            return

        self._memory.append(message_data)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        merged = 0
        deduped = 0
        seen_replies: set[str] = set()
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if not (isinstance(content, str) and content):
                logger.warning(f"Skipping invalid message from history: {msg}")
                continue
            # Same collapse as the live path: a stored history is full of user
            # turns whose replies were interrupted, and replaying them as
            # separate questions makes the model answer a stale one.
            # Drop contentless assistant turns (bare "..." left by interrupts).
            # Replayed as context they teach the model that an empty reply is a
            # valid answer, and it starts producing them on its own.
            if role == "assistant" and not re.sub(r"[.…\s]", "", content):
                continue
            # Drop REPEATED identical assistant replies. Once the model has said
            # the same thing twice (classically its own job description), those
            # copies act as few-shot examples and it says it again for every
            # unclear input — "ずっと自己紹介する". Keeping the first occurrence
            # preserves the information without reinforcing the loop.
            if role == "assistant":
                _norm = " ".join(content.split()).lower()
                if _norm in seen_replies:
                    deduped += 1
                    continue
                seen_replies.add(_norm)
            if role == "user" and self._memory and self._memory[-1]["role"] == "user":
                self._memory[-1]["content"] = _merge_user_turns(
                    self._memory[-1]["content"], content
                )
                merged += 1
                continue
            self._memory.append({"role": role, "content": content})
        logger.info(
            f"Loaded {len(self._memory)} messages from history"
            + (f" ({merged} unanswered user turns merged)" if merged else "")
            + (f" ({deduped} repeated replies dropped)" if deduped else "")
        )

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        # A barge-in that lands before ANY audio was heard leaves heard_response
        # empty, and the old code still wrote `heard_response + "..."` — i.e. a
        # bare "..." — as the assistant's turn. Worse, when _memory[-1] was the
        # PREVIOUS turn's good reply, that reply was overwritten with it. Nothing
        # was said, so there is nothing to record: leave the trailing assistant
        # turn alone and let the "[Interrupted by user]" marker below carry the
        # fact of the interrupt on its own. (The two branches of the old
        # `endswith("...")` test were byte-identical, so the guard never did
        # anything.) See _add_message for why a bare "..." is load-bearing.
        spoken = (heard_response or "").strip()
        if spoken:
            if self._memory and self._memory[-1]["role"] == "assistant":
                self._memory[-1]["content"] = spoken + "..."
            else:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": spoken + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []

        while True:
            stream = self._llm.chat_completion(messages, self._system, tools=tools)
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    yield text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = self._system
        # Cap the tool ROUNDS: the model paged fetch_content 800 chars at a time
        # (5-6 rounds x an LLM round-trip each = ~25-30s of dead air before the
        # spoken answer; the user then talks into the silence and the barge-in
        # kills the reply). After the cap the tools are withheld, forcing the
        # model to ANSWER with what it has. Tune: LLM_TOOL_MAX_ROUNDS.
        tool_rounds = 0
        max_tool_rounds = int(os.getenv("LLM_TOOL_MAX_ROUNDS", "3"))

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{self._system}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = self._system
                tools_for_api = None
            else:
                current_system_prompt = self._system
                tools_for_api = tools
                if tool_rounds >= max_tool_rounds:
                    # budget spent: no more tools — answer from gathered results.
                    # Steer explicitly: without this the model (mid tool-groove)
                    # emits a RAW <tool_call> JSON as plain text, which then got
                    # SPOKEN by TTS.
                    logger.info(
                        f"[tool-cap] {tool_rounds} tool rounds used — forcing answer"
                    )
                    tools_for_api = None
                    _cap_notice = _TOOL_CAP_NOTICE.get(
                        self._language, _TOOL_CAP_NOTICE["ja"]
                    )
                    if not any(
                        m.get("content") == _cap_notice for m in messages[-2:]
                    ):
                        messages.append({"role": "system", "content": _cap_notice})

            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False
            suppress_buf = ""  # partial-tag carry for the <tool_call> text filter
            in_tool_txt = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                        # Suppress RAW "<tool_call>...</tool_call>" leaking as
                        # text (hermes format; happens when tools are withheld
                        # after the round cap) — it must never reach TTS.
                        emit, suppress_buf, in_tool_txt = _filter_tool_call_text(
                            event, suppress_buf, in_tool_txt
                        )
                        if emit:
                            yield emit
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            # content is deliberately None even when the model
                            # streamed a pre-amble before its tool call. Sending
                            # that pre-amble back ("I am retrieving the latest
                            # Japanese news…") makes the model treat the turn as
                            # already answered, so after the search results
                            # arrive it returns NOTHING and the user only ever
                            # hears the promise. The pre-amble was already
                            # spoken; the model only needs the call + result.
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_rounds += 1
                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            messages = self._to_messages(input_data)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                token_stream = self._llm.chat_completion(messages, self._system)
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output
        # Turn is fully generated and streamed — the user already heard it, so
        # compaction here is OFF the response critical path. Fire-and-forget:
        # if the history is near the context cap, summarize the old turns in the
        # background so the NEXT turn starts from a compact memory instead of
        # having oldest turns hard-dropped (which lost context = "同じことを言い
        # 続ける"). Never blocks; the send-time trim stays as the safety net.
        self.schedule_compaction_if_needed()

    # ---- background memory compaction ------------------------------------- #
    # Estimated-token threshold on self._memory ALONE (system prompt + the
    # generated answer live outside it) past which we summarize. Kept well under
    # the model's 16384 so a summary + recent turns + system + answer all fit.
    _COMPACT_THRESHOLD_TOKENS = int(os.getenv("LLM_COMPACT_AT", "8000"))
    _COMPACT_KEEP_RECENT = int(os.getenv("LLM_COMPACT_KEEP", "6"))

    @staticmethod
    def _mem_tokens(memory: List[Dict[str, Any]]) -> int:
        """Conservative token estimate (JA/tool-dense ~= chars/1.3)."""
        n = 0
        for m in memory:
            c = m.get("content")
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                n += sum(len(str(p.get("text", ""))) for p in c if isinstance(p, dict))
        return int(n / 1.3)

    def schedule_compaction_if_needed(self) -> None:
        """If memory is near the cap, kick off a background summarization. Cheap
        guard so it's safe to call every turn; the real work runs in a task."""
        if self._compacting:
            return
        if self._mem_tokens(self._memory) < self._COMPACT_THRESHOLD_TOKENS:
            return
        if len(self._memory) <= self._COMPACT_KEEP_RECENT + 4:
            return  # too little history to be worth summarizing
        try:
            import asyncio

            asyncio.get_running_loop().create_task(self._run_compaction())
        except RuntimeError:
            pass  # no loop (sync context) — skip; send-time trim still protects

    async def _run_compaction(self) -> None:
        """Summarize the OLD prefix of memory into one system note, preserving
        the most recent turns verbatim. Interrupt-safe: we snapshot the prefix,
        summarize it off-loop, then splice ONLY that prefix out — any turns that
        arrived meanwhile (memory tail) are untouched."""
        if self._compacting:
            return
        self._compacting = True
        try:
            n = len(self._memory)
            keep = self._COMPACT_KEEP_RECENT
            if n <= keep + 4:
                return
            old = self._memory[: n - keep]  # snapshot prefix to summarize
            old_len = len(old)
            summary = await self._summarize(old)
            if not summary:
                return
            # Splice: replace exactly the first old_len entries (unchanged since
            # the snapshot — new turns only ever append to the tail) with one
            # summary note. If a prior summary note is at [0], fold it in.
            note_label = (
                "[Summary of the conversation so far]"
                if self._language == "en"
                else "[これまでの会話の要約]"
            )
            note = {
                "role": "system",
                "content": note_label + "\n" + summary,
            }
            self._memory = [note] + self._memory[old_len:]
            logger.info(
                f"[compact] summarized {old_len} old messages -> 1 note "
                f"(memory now {len(self._memory)} msgs, "
                f"~{self._mem_tokens(self._memory)} tok)"
            )
        except Exception as e:  # noqa: BLE001 — compaction must never break chat
            logger.warning(f"[compact] failed: {e!r}")
        finally:
            self._compacting = False

    async def _summarize(self, msgs: List[Dict[str, Any]]) -> str:
        """One-shot LLM summary of a slice of conversation. Uses the same LLM
        as chat (background task, so it doesn't gate any reply)."""
        convo = []
        for m in msgs:
            role = m.get("role")
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(
                    str(p.get("text", "")) for p in c if isinstance(p, dict)
                )
            en = self._language == "en"
            if not c:
                if m.get("tool_calls"):
                    c = "(tool call)" if en else "(ツール呼び出し)"
                else:
                    continue
            if en:
                who = {"user": "User", "assistant": "Assistant"}.get(role, role)
            else:
                who = {"user": "ユーザー", "assistant": "アシスタント"}.get(role, role)
            convo.append(f"{who}: {c}")
        transcript = "\n".join(convo)
        if self._language == "en":
            sys_prompt = (
                "You are a conversation-log compressor. Summarize the following "
                "conversation as concise bullet points usable as future context. "
                "Keep the user's preferences, names, decisions, ongoing topics "
                "and important facts; drop repetitive small talk. English, under "
                "200 words. Output the summary only."
            )
        else:
            sys_prompt = (
                "あなたは会話ログを圧縮する要約器です。以下の会話を、後で文脈として"
                "使えるように簡潔な箇条書きで要約してください。ユーザーの好み・名前・"
                "決定事項・進行中の話題・重要な事実を保持し、雑談の繰り返しは省く。"
                "日本語で、200語以内。要約のみ出力。"
            )
        out = ""
        try:
            async for chunk in self._llm.chat_completion(
                messages=[{"role": "user", "content": transcript}],
                system=sys_prompt,
            ):
                if isinstance(chunk, str):
                    out += chunk
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[compact] summarize LLM call failed: {e!r}")
            return ""
        return out.strip()

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
