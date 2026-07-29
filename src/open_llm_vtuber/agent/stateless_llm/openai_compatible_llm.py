"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

import os
import time
import httpx
from typing import AsyncIterator, List, Dict, Any
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface
from ...mcpp.types import ToolCallObject


def _est_tokens(messages: List[Dict[str, Any]], system: str | None) -> int:
    """CONSERVATIVE input-token estimate (overestimate = safe for a cap). JA
    text tokenizes ~1 token/char and tool-result blobs full of digits/symbols
    (weather tables, nav menus) tokenize even denser than plain prose, so the
    earlier chars/2.2 under-counted badly (est 13k vs real 20k) and the trim
    never fired. chars/1.3 tracks the real vLLM count closely on this traffic."""
    n = len(system or "")
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):  # vision/content-part messages
            n += sum(len(str(p.get("text", ""))) for p in c if isinstance(p, dict))
        for tc in m.get("tool_calls") or []:
            n += len(str(tc))
    return int(n / 1.3)


# Hard per-message content cap (chars) for OLD messages. A single fetch_content
# can return an 8000-char page; several in one loop overflow the context. But
# the CURRENT turn's tool results ARE the answer ("調べた内容") — capping them
# dropped the very data the user asked for. So the newest _PROTECT_RECENT
# messages are NEVER capped; only older history is. Override: LLM_MSG_CAP.
_MSG_CAP_CHARS = int(os.getenv("LLM_MSG_CAP", "3000"))
# How many trailing messages to keep verbatim (never cap): the current tool
# exchange = interleaved assistant(tool_calls)+tool results, several messages.
_PROTECT_RECENT = int(os.getenv("LLM_PROTECT_RECENT", "12"))


def _cap_message_bodies(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shallow copy with over-long OLD message bodies truncated. The newest
    _PROTECT_RECENT messages (the live tool exchange) are left intact so the
    freshly-fetched content always reaches the model in full."""
    cutoff = max(0, len(messages) - _PROTECT_RECENT)
    out = []
    for i, m in enumerate(messages):
        c = m.get("content")
        if i < cutoff and isinstance(c, str) and len(c) > _MSG_CAP_CHARS:
            logger.info(
                f"[context-trim] capped OLD {m.get('role')} message "
                f"({len(c)}->{_MSG_CAP_CHARS} chars)"
            )
            m = {**m, "content": c[:_MSG_CAP_CHARS] + " …[truncated]"}
        out.append(m)
    return out


def _trim_to_context(
    messages: List[Dict[str, Any]], system: str | None, budget_tokens: int
) -> List[Dict[str, Any]]:
    """Drop the OLDEST turns until the estimated input fits budget_tokens.

    A long chat history + accumulated tool results (each fetch_content adds a
    big web-page blob) blew past vLLM's max context (16384) -> HTTP 502 -> the
    reply never generated ("ツールを使うとしゃべらない"). Keep the newest
    messages, but never split a tool_calls assistant from its tool results (an
    orphaned tool message is a 400). The newest message is ALWAYS kept — the
    system prompt (Rem persona is 24.5KB!) can exceed the whole budget by
    itself, and that case must degrade to tail-only, not crash.
    """
    sys_est = int(len(system or "") / 1.3)
    msg_budget = max(2500, budget_tokens - sys_est)
    if _est_tokens(messages, None) <= msg_budget:
        return messages
    # Walk from the newest, keeping a growing suffix that fits.
    keep_from = len(messages) - 1  # floor: always keep the newest message
    for i in range(len(messages) - 1, -1, -1):
        if _est_tokens(messages[i:], None) > msg_budget:
            keep_from = min(i + 1, len(messages) - 1)
            break
        keep_from = i
    # If the window starts on orphan tool results, EXTEND BACKWARD to include
    # their owning assistant(tool_calls) instead of discarding them — dropping
    # them loses the freshly fetched content, which is exactly what the user
    # asked about ("調べてきた内容がスキップされる"). A small budget overshoot
    # is fine: the estimate is conservative and generation headroom is ~4k.
    # (Starting ON an assistant with tool_calls is valid — its results follow.)
    while 0 < keep_from < len(messages) and messages[keep_from].get("role") == "tool":
        keep_from -= 1  # walk back to the assistant that owns these results
    trimmed = messages[keep_from:]
    dropped = keep_from
    if dropped:
        logger.warning(
            f"[context-trim] dropped {dropped} oldest message(s) to fit msg-budget "
            f"~{msg_budget} tok (msgs were ~{_est_tokens(messages, None)}, "
            f"system est ~{sys_est})"
        )
    return trimmed or messages[-1:]


class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        # Public-gateway fallback: the primary base_url is usually the SSH LAN
        # tunnel (127.0.0.1:13000). When the tunnel is down a conversation used
        # to just DIE ("Failed to connect to the LLM API"); instead retry the
        # turn against the public gateway (same Bearer key, +~0.35s/req).
        # LLM_BASE_FALLBACK overrides; empty string disables.
        fallback = os.getenv("LLM_BASE_FALLBACK", "https://gpu.aiccore-uno.ai/v1")
        self._fallback_base = fallback if fallback != base_url else ""
        self._fallback_client: AsyncOpenAI | None = None
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
            # With a fallback available the primary must FAIL FAST: the SDK's
            # default 2 connect-retries + backoff burned ~7s per turn against a
            # dead tunnel before the fallback ever fired (measured TTFA
            # llm_first_sentence 7.7-8.4s). 0 retries when we have a plan B.
            max_retries=0 if self._fallback_base else 2,
            # SHORT timeout on the primary (tunnel): a half-dead tunnel (TCP
            # connects, PC-B silent after a VPN blip) must trip the sticky
            # fallback in seconds, not hang on the SDK's 600s default. vLLM TTFT
            # is ~0.4s so a 12s read ceiling never clips a healthy stream.
            timeout=(
                httpx.Timeout(12.0, connect=3.0)
                if self._fallback_base
                else httpx.Timeout(600.0, connect=10.0)
            ),
        )
        self.support_tools = True
        # While the primary is known-down, go STRAIGHT to the fallback for this
        # many seconds (skip even the fast-fail probe), then re-probe.
        self._primary_down_until = 0.0
        if self._fallback_base:
            self._fallback_client = AsyncOpenAI(
                base_url=self._fallback_base,
                organization=organization_id,
                project=project_id,
                api_key=llm_api_key,
            )

        logger.info(
            f"Initialized AsyncLLM with the parameters: {self.base_url}, {self.model}"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.

        Yields:
        - str: The content of each chunk from the API response.
        - List[ChoiceDeltaToolCall]: The tool calls detected in the response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        stream = None
        # Tool call related state variables
        accumulated_tool_calls = {}
        in_tool_call = False

        try:
            # Fit the context window: vLLM caps input at max_model_len (16384
            # here); history + accumulated tool blobs overflow it -> 502 and no
            # reply. Two stages: (1) cap each oversized message body (the tool
            # blobs that dominate a single tool turn), then (2) drop oldest
            # turns to the token budget. Override via LLM_MSG_CAP / LLM_CTX_BUDGET.
            budget = int(os.getenv("LLM_CTX_BUDGET", "12000"))
            messages = _cap_message_bodies(messages)
            _tool_before = sum(1 for m in messages if m.get("role") == "tool")
            messages = _trim_to_context(messages, system, budget)
            _tool_after = sum(1 for m in messages if m.get("role") == "tool")
            if _tool_before and _tool_after < _tool_before:
                logger.warning(
                    f"[context-trim] DROPPED {_tool_before - _tool_after}/"
                    f"{_tool_before} tool results — raise LLM_CTX_BUDGET or the "
                    f"answer loses fetched content"
                )
            elif _tool_after:
                logger.info(f"[context] {_tool_after} tool result(s) reaching LLM")

            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]
            logger.debug(f"Messages: {messages_with_system}")

            available_tools = tools if self.support_tools else NOT_GIVEN

            request_kwargs = dict(
                messages=messages_with_system,
                model=self.model,
                stream=True,
                # Final chunk carries usage -> log REAL prompt_tokens (the char
                # estimate drifts by content type; this calibrates the budget).
                stream_options={"include_usage": True},
                temperature=self.temperature,
                tools=available_tools,
            )
            # Sticky fallback: while the tunnel is known-down, don't pay even
            # the probe — go straight to the public gateway. After the window
            # expires one turn re-probes the primary (fail-fast, max_retries=0).
            primary = self.client
            if (
                self._fallback_client is not None
                and time.monotonic() < self._primary_down_until
            ):
                primary = self._fallback_client
            try:
                stream: AsyncStream[
                    ChatCompletionChunk
                ] = await primary.chat.completions.create(**request_kwargs)
            except APIConnectionError as e:
                if (
                    self._fallback_client is None
                    or primary is self._fallback_client
                ):
                    raise
                self._primary_down_until = time.monotonic() + 300.0
                logger.warning(
                    f"[llm] {self.base_url} unreachable ({e.__class__.__name__}) "
                    f"— falling back to {self._fallback_base} (sticky 300s)"
                )
                stream = await self._fallback_client.chat.completions.create(
                    **request_kwargs
                )
            logger.debug(
                f"Tool Support: {self.support_tools}, Available tools: {available_tools}"
            )

            async for chunk in stream:
                # usage rides a trailing chunk with EMPTY choices — read it
                # before the choices guard below skips it.
                if getattr(chunk, "usage", None) and chunk.usage.prompt_tokens:
                    logger.info(
                        f"[context] REAL prompt_tokens={chunk.usage.prompt_tokens} "
                        f"(vLLM cap {os.getenv('LLM_CTX_MAX', '24576')})"
                    )
                # Guard against chunks with missing choices field (e.g., from OpenWebUI)
                if not chunk.choices:
                    continue

                if self.support_tools:
                    has_tool_calls = (
                        hasattr(chunk.choices[0].delta, "tool_calls")
                        and chunk.choices[0].delta.tool_calls
                    )

                    if has_tool_calls:
                        logger.debug(
                            f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}"
                        )
                        in_tool_call = True
                        # Process tool calls in the current chunk
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            index = (
                                tool_call.index if hasattr(tool_call, "index") else 0
                            )

                            # Initialize tool call for this index if needed
                            if index not in accumulated_tool_calls:
                                accumulated_tool_calls[index] = {
                                    "index": index,
                                    "id": getattr(tool_call, "id", None),
                                    "type": getattr(tool_call, "type", None),
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Update tool call information
                            if hasattr(tool_call, "id") and tool_call.id:
                                accumulated_tool_calls[index]["id"] = tool_call.id
                            if hasattr(tool_call, "type") and tool_call.type:
                                accumulated_tool_calls[index]["type"] = tool_call.type

                            # Update function information
                            if hasattr(tool_call, "function"):
                                if (
                                    hasattr(tool_call.function, "name")
                                    and tool_call.function.name
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "name"
                                    ] = tool_call.function.name
                                if (
                                    hasattr(tool_call.function, "arguments")
                                    and tool_call.function.arguments
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "arguments"
                                    ] += tool_call.function.arguments

                        continue

                    # If we were in a tool call but now we're not, yield the tool call result
                    elif in_tool_call and not has_tool_calls:
                        in_tool_call = False
                        # Convert accumulated tool calls to the required format and output
                        logger.info(f"Complete tool calls: {accumulated_tool_calls}")

                        # Use the from_dict method to create a ToolCallObject instance from a dictionary
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]

                        yield complete_tool_calls
                        accumulated_tool_calls = {}  # Reset for potential future tool calls

                # Process regular content chunks
                if len(chunk.choices) == 0:
                    logger.info("Empty chunk received")
                    continue
                elif chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content

            # If stream ends while still in a tool call, make sure to yield the tool call
            if in_tool_call and accumulated_tool_calls:
                logger.info(f"Final tool call at stream end: {accumulated_tool_calls}")

                # Create a ToolCallObject instance from a dictionary using the from_dict method.
                complete_tool_calls = [
                    ToolCallObject.from_dict(tool_data)
                    for tool_data in accumulated_tool_calls.values()
                ]

                yield complete_tool_calls

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
            if "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Messages: {messages}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")
