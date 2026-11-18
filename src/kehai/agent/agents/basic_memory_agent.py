import os
import re
from datetime import datetime
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
from ..stateless_llm.openai_compatible_llm import (
    AsyncLLM as OpenAICompatibleAsyncLLM,
    _safe_cal,
)
from ...chat_history_manager import get_history
from ...conversations.speculative import SPOKEN_PLACEHOLDER
from ...utils.sentence_spacing import unglue
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
    normalize_labelled_tags,
    strip_labelled_tags,
    cut_repeat,
    _GESTURE_TAG,
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
# Sent once when a round ends on a promised lookup with no tool call
# (is_announce_only). The Thinker said "I can check that for you." and stopped,
# 2026-11-04 19:13; asked again with this line it called search / time in 3 of 3.
_TOOL_NUDGE = {
    "en": "You said you would check. Call the right tool now instead of replying in text.",
    "ja": "確認すると言いました。文章で答えずに、今すぐ適切なツールを呼んでください。",
}

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


_TIME_LINE_PREFIX = "Current local time:"
_INTERRUPT_MARK = "[Interrupted by user]"

# A reply that only says she did not understand. Kept in context, each one is a
# few-shot example of the answer shape, and the model (greedy on vLLM-Omni until
# 2026-11-04) gives it to everything after a handful: 294 of 324 replies on
# 2026-11-04 afternoon were "I am not sure what you mean. Could you clarify
# your question?", and the same audio answered normally with no context.
_NON_ANSWER_RE = re.compile(
    r"\b(?:"
    r"(?:i(?:'m| am)|i'm) not (?:sure|certain|clear) (?:what|about what|who|which)"
    r"|(?:i(?:'m| am)) not (?:aware|sure) (?:of )?what (?:exactly )?you (?:would like|want|mean|are (?:asking|referring))"
    # Refusals that only say she does not know. As examples they taught her to
    # answer "I am not aware of tomorrow's weather." instead of searching
    # (2026-11-04 19:02 and again at 19:08 for the same question).
    r"|(?:i(?:'m| am)) not (?:aware|familiar|informed) (?:of|with|about)"
    r"|i (?:do not|don't) have (?:any )?(?:specific )?(?:information|details|data|access)"
    r"|(?:i(?:'m| am)) (?:unable|not able) to (?:browse|access|search|look|find|provide|check)"
    r"|(?:could|can|would) you (?:please )?(?:clarify|rephrase|repeat|say that again|elaborate|explain what you mean)"
    r"|please (?:clarify|rephrase|repeat)"
    r"|i (?:did not|didn't|could not|couldn't) (?:quite )?(?:catch|understand|hear|follow|get) (?:that|you|what|it)"
    r"|i (?:do not|don't) (?:quite )?(?:understand|follow)"
    r"|(?:that|this|it) (?:doesn't|does not) (?:quite )?make (?:any )?sense"
    r"|you(?:'re| are) not making (?:any |much )?sense"
    r"|what do you mean"
    r"|sorry,? (?:what|i missed that|come again)"
    r"|(?:i )?beg your pardon"
    r")",
    re.IGNORECASE,
)


def _reply_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text or "")
    return [s.strip() for s in re.split(r"[.?!…。？！]+", text) if re.search(r"\w", s)]


def is_non_answer(text: str) -> bool:
    """True when every sentence of a reply is "I did not understand" talk."""
    sents = _reply_sentences(text)
    return bool(sents) and all(_NON_ANSWER_RE.search(s) for s in sents)


# A reply that only asks back what the visitor meant ("Are you asking about the
# South Market?", "Are you referring to a probability puzzle?"). One is fine — the
# persona allows a single question when she could not tell — but kept in context
# they became the answer to everything: 67 of her 157 replies in the Iris history
# by 2026-11-07, whatever the visitor said, the last 30 nearly all of them. Only
# the latest stays in context (the visitor's "yes" needs it); older ones go with
# their question, as non-answers do (tidy_memory).
_ECHO_Q_RE = re.compile(
    r"^\W*(?:are you (?:asking|referring|talking|saying|confirming|inquiring|wondering|looking|trying to say)"
    r"|do you mean|did you mean|you mean|is that what you|were you (?:asking|referring|saying))\b",
    re.IGNORECASE,
)


# "Could you please clarify what you'd like me to explain?" — the request that
# follows the echo question (2026-11-09: three "Are you referring to X? Could you
# please clarify ...?" replies in a row stayed in the context, because only bare
# echo questions were recognised).
_CLARIFY_RE = re.compile(
    r"\b(?:clarify|seems (?:incomplete|unclear|cut off)|(?:was|is|got) (?:cut off|unclear|lost)"
    r"|did(?:n't| not) (?:quite )?(?:catch|hear|get|understand)|could you (?:please )?(?:specify|elaborate|be more specific|rephrase|repeat)"
    r"|what (?:exactly )?(?:do|did) you mean|(?:which|what) (?:part|aspect|one|kind)\b.*\byou(?:'d| would)? (?:like|want|mean))",
    re.IGNORECASE,
)


def is_echo_question(text: str) -> bool:
    """True when the reply only asks back: "did you mean ...?" / "could you
    clarify ...?" questions, nothing answered."""
    t = re.sub(r"\[[^\]]*\]", " ", text or "").strip()
    sents = [x.strip() for x in re.split(r"(?<=[.?!])\s+", t) if re.search(r"\w", x)]
    return (
        bool(sents) and len(sents) <= 3
        and all(x.endswith("?") or (_CLARIFY_RE.search(x) and len(x.split()) <= 8) for x in sents)
        and any(_ECHO_Q_RE.search(x) or _CLARIFY_RE.search(x) for x in sents)
    )


# A reply that OPENS by asking the visitor what they meant (2026-11-09 03:10:
# with the persona telling her not to, "Are you asking for a recommendation in
# Barro Anime? If so, could you please clarify your request?" and "I did not
# understand the question. Could you please rephrase?" still came out — the
# 7B-class Thinker follows the context and its habit more than the rule). Caught
# on its first words, before anything is spoken, and asked again once.
_ASKBACK_GUARD = os.getenv("ASKBACK_GUARD", "1") == "1"
_ASKBACK_OPEN_RE = re.compile(
    r"^\W*(?:are you (?:asking|referring|talking|saying|inquiring|wondering|looking for|trying to)"
    r"|do you mean|did you mean|you mean|is that what you|were you (?:asking|referring|saying)"
    r"|i (?:did not|didn't|do not|don't|am not sure|'m not sure|could not|couldn't) (?:quite )?"
    r"(?:understand|catch|follow|get|hear|know what you|sure what you)"
    r"|i am not sure what you|i'm not sure what you|i am unsure|i'm unsure|i am not certain what"
    r"|(?:could|can|would) you (?:please )?(?:clarify|rephrase|repeat|specify|elaborate|say that again)"
    r"|(?:the|your|that) (?:sentence|question|request|message) (?:seems|is|was|sounds) (?:incomplete|unclear|cut))",
    re.IGNORECASE,
)
# Put on the visitor's own turn for the retry. Measured against a trailing system
# message (Thinker, Iris persona, replies that had asked back, x3 each): asked back
# again 29/69 vs 22/69 on mechanically broken public MASSIVE questions, 3/24 vs
# 17/24 on the real 11-09 turns — the system message barely moved real speech.
_ASKBACK_STARTS = {"iris", "are", "do", "did", "you", "is", "were", "i", "i'm", "could", "can", "would", "the", "your", "that"}
_ASKBACK_NUDGE = "(Answer what I most likely meant, directly. Do not ask me what I meant.)"


_IFSO_RE = re.compile(r"^\s*(?:if so|if that is the case|if that's the case|if you mean[^,.?!]{0,60}|if that is what you mean)\s*,\s*", re.IGNORECASE)


def strip_askback(text: str) -> str | None:
    """For the retried reply: "Are you asking about X? If so, <answer>" -> "<answer>".
    None = not that shape yet / not that shape (speak it as it is)."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip(), maxsplit=1)
    if len(parts) < 2 or not askback_opening(parts[0] + " "):
        return None
    m = _IFSO_RE.match(parts[1])
    if not m:
        return None
    rest = parts[1][m.end():]
    return rest[:1].upper() + rest[1:] if rest else None


_REPEAT_WORDS = 6
_REPEAT_NUDGE = "(Answer this, not what I asked before. Do not repeat what you already told me.)"


def _norm_words(text: str) -> list[str]:
    t = re.sub(r"\[[^\]]*\]", " ", text or "")
    t = re.sub(r"^\W*iris\s*:\s*", "", t, flags=re.IGNORECASE)
    return re.findall(r"[a-z0-9']+", t.lower())


def repeat_opening(text: str, recent: list[list[str]], final: bool = False) -> bool | None:
    """Does the reply open with the same words as one of her recent replies?
    None = still the start of one of them, too short to tell. (2026-11-09 03:31:
    "I am Iris, the android guide for this maker space." opened 4 replies in a
    row, to "Hi", "They didn't know who made it" and "explain about that team".)"""
    w = _norm_words(text)
    if not w or not recent:
        return False
    complete, last = w[:-1], w[-1]
    if final:
        complete, last = w, ""
    live = [r for r in recent if r[:len(complete)] == complete and (not last or (len(r) > len(complete) and r[len(complete)].startswith(last)))]
    if not live:
        return False
    if len(w) >= _REPEAT_WORDS:
        return True
    return None if not final else False


def askback_opening(text: str) -> bool | None:
    """Does the reply so far open by asking back? None = too few words to tell."""
    t = re.sub(r"\[[^\]]*\]", " ", text or "")
    t = re.sub(r"^\W*iris\s*:\s*", "", t, flags=re.IGNORECASE)  # a speaker label she sometimes writes
    first = re.split(r"(?<=[.?!])\s", t.strip(), maxsplit=1)[0]
    words = re.findall(r"[\w']+", t.lower())
    if words and words[0] not in _ASKBACK_STARTS:
        return False  # cannot be one: speak at once
    # Wait for the whole opening (2026-11-09 03:24: decided on 5 tokens, "I am not
    # sure what" did not match yet and "... you mean. Could you please rephrase or
    # clarify?" was spoken).
    if len(words) < 8 and not re.search(r"[.?!]\s*$", t.strip()):
        return None
    return bool(_ASKBACK_OPEN_RE.search(first))


# The general case of the above (2026-11-08): ANY reply shape that piles up in the
# context becomes her answer to everything — "I am not sure what you mean" on
# 11-04, "Are you referring to ...?" on 11-07, "Hello. I am functioning
# optimally." x9 in the same history. Measured on clear public questions with the
# Iris persona: 0 such replies in the context -> 0/60 echo replies, 1 -> 5/60,
# 3 -> 8/60, 5 -> 27/60, 12 -> 59/60. A word list only catches the shapes seen so
# far, so the context also keeps at most _SHAPE_KEEP replies that open with the
# same words (the latest ones); older ones go with their question.
_SHAPE_WORDS = 3
_SHAPE_KEEP = 2


def _reply_shape(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", str(text or "")).lower()
    return " ".join(re.findall(r"[\w']+", t)[:_SHAPE_WORDS])


def _norm_reply(text: str) -> str:
    """Comparison key for replies: tags, emoji, punctuation and a trailing
    "..." (an interrupted copy) do not make a reply different."""
    t = re.sub(r"\[[^\]]*\]", " ", str(text or "")).lower()
    return " ".join(re.sub(r"[^\w']+", " ", t).split())


_LOOKUP_RE = re.compile(r"\b(retriev\w*|search\w*|look\w* (?:it |that |this )?up|fetch\w*|check\w*|find out)\b", re.I)
_REFUSAL_RE = re.compile(r"\b(cannot|can't|can not|unable|do(?:es)? not have|don't have|no access|not able)\b", re.I)


def is_announce_only(text: str) -> bool:
    """A reply that PROMISES a lookup and delivers nothing (the history rule in
    single_conversation, here for the model's context): short, a lookup verb, no
    digits, not a refusal. Only meaningful when no tool was called that turn.
    In context they taught "I can check the weather forecast for tomorrow."
    instead of calling search (2026-11-04 19:09; the same question with no
    history called search every time)."""
    t = (text or "").strip()
    if not t or _REFUSAL_RE.search(t) or re.search(r"\d", t) or len(t.split()) >= 20:
        return False
    return bool(_LOOKUP_RE.search(t)) or bool(re.search(r"[—–:：]\s*$", t))


def _near_repeat(key: str, earlier: list[str]) -> bool:
    """Same reply, or the same reply with a few words added or dropped: one
    key's words (5 or more) almost all inside an earlier one's. "Interesting. I
    am not capable of physical movements such as raising my eyebrows." and the
    same without "Interesting." both stayed in context as different replies and
    were copied to unrelated questions (2026-11-04 18:59)."""
    if not key:
        return False
    words = set(key.split())
    for e in earlier:
        if e == key:
            return True
        ew = set(e.split())
        small = min(len(words), len(ew))
        if small >= 5 and len(words & ew) / small >= 0.85:
            return True
    return False


def _is_time_line(m: dict) -> bool:
    return m.get("role") == "system" and str(m.get("content", "")).startswith(_TIME_LINE_PREFIX)


def _is_marker(m: dict) -> bool:
    return m.get("role") in ("system", "user") and m.get("content") == _INTERRUPT_MARK


def _dump_omni_turn(audio_b64: str, text: str, memory: list) -> None:
    """OMNI_AUDIO_DUMP=<dir>: keep what the Thinker is given for a turn — the
    audio (wav) and the tail of the context (json: the last 8 messages, text
    truncated) — newest 60 turns. For "she says my words make no sense" the
    question is whether the audio she got was my words, and what preceded them;
    this answers both offline (replay the wav alone / with the context).
    2026-11-04. Review material only; never on the reply path for long."""
    d = os.getenv("OMNI_AUDIO_DUMP", "")
    if not d:
        return
    try:
        import base64 as _b64
        import glob as _glob
        import json as _json
        import time as _time

        os.makedirs(d, exist_ok=True)
        stamp = _time.strftime("%Y%m%d-%H%M%S") + f".{int((_time.time() % 1) * 1000):03d}"
        with open(os.path.join(d, f"{stamp}.wav"), "wb") as f:
            f.write(_b64.b64decode(audio_b64))
        tail = []
        for m in memory[-8:]:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
            tail.append({"role": m.get("role"), "content": str(c)[:160]})
        with open(os.path.join(d, f"{stamp}.json"), "w", encoding="utf-8") as f:
            _json.dump({"asr_text": text, "context_tail": tail, "memory_len": len(memory)}, f, ensure_ascii=False, indent=1)
        old = sorted(_glob.glob(os.path.join(d, "*.wav")))
        for path in old[: max(0, len(old) - 60)]:
            for ext in (".wav", ".json"):
                try:
                    os.remove(path[:-4] + ext)
                except OSError:
                    pass
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[omni-dump] {e!r}")


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
        self._turn_dropped = False
        self._turn_used_tool = False
        self._nudged = False
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
        # The model's own glued sentences ("fine.Same as always") must not
        # become the next request's example — see sentence_spacing.
        text_content = unglue(text_content)

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
        # Live counterpart of the load-time rule in set_memory_from_history: a
        # reply identical to one already in memory is not kept. Three
        # "You're not making sense again. What are you trying to say?" in a row
        # (2026-11-04 05:45) were each a few-shot example for the next; the
        # history is still written, so nothing is lost on disk.
        if role == "assistant" and text_content:
            _norm = _norm_reply(text_content)
            why = None
            if is_non_answer(text_content):
                why = "non-answer"
            elif not getattr(self, "_turn_used_tool", False) and is_announce_only(text_content):
                why = "announce-only (no tool was called)"
            elif _near_repeat(_norm, [_norm_reply(m.get("content", "")) for m in self._memory if m.get("role") == "assistant"]):
                why = "repeated reply"
            if why:
                # The WHOLE turn leaves the context — its question and the
                # question's time line too. Leaving the question made it
                # "unanswered" and glued to the next one; leaving the time line
                # piled up 346 orphan clock lines in one afternoon (2026-11-04).
                # handle_interrupt then records nothing for this turn either.
                self._drop_open_turn()
                self._turn_dropped = True
                logger.info(f"[memory] {why} (and its turn) not kept in context: {text_content[:60]!r}")
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
        if role == "assistant":
            self.schedule_post_reply_warm()

    def _drop_open_turn(self) -> None:
        """Remove the newest turn's question and its time line (the reply, if
        any, is the caller's to remove first)."""
        if self._memory and self._memory[-1].get("role") == "user":
            self._memory.pop()
        if self._memory and _is_time_line(self._memory[-1]):
            self._memory.pop()

    def tidy_memory(self) -> int:
        """Keep the context in the shape [time] user assistant [marker], turn by
        turn. Removes, IN PLACE (speculative.discard tracks the list by
        identity): turns answered by a non-answer or by a reply already given,
        time lines with no question after them, interrupt markers with no reply
        before them. Idempotent; returns how many messages went.

        Why (2026-11-04 afternoon): the request held 693 messages — 346 clock
        lines, 160 interrupt markers, 135 "I am not sure what you mean" replies
        and 36 user turns. The repeated-reply rule removed questions but left
        their clock lines; the interrupt handler then wrote the heard copy of
        the dropped reply back with a marker. Every reply after that was the
        same non-answer."""
        mem = self._memory
        drop: set[int] = set()
        seen: list[str] = []
        last_reply = max((i for i, m in enumerate(mem) if m.get("role") == "assistant"), default=-1)
        # replies whose opening words already appear _SHAPE_KEEP times LATER on
        shape_n: dict[str, int] = {}
        over_shape: set[int] = set()
        for i in range(len(mem) - 1, -1, -1):
            m = mem[i]
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                k = _reply_shape(m["content"])
                if k:
                    shape_n[k] = shape_n.get(k, 0) + 1
                    if shape_n[k] > _SHAPE_KEEP:
                        over_shape.add(i)
        for i, m in enumerate(mem):
            if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
                continue
            key = _norm_reply(m["content"])
            if is_non_answer(m["content"]) or _near_repeat(key, seen) or i in over_shape or (
                i != last_reply and is_echo_question(m["content"])
            ):
                drop.add(i)
                j = i - 1
                if j >= 0 and mem[j].get("role") == "user":
                    drop.add(j)
                    j -= 1
                if j >= 0 and _is_time_line(mem[j]):
                    drop.add(j)
                k = i + 1
                while k < len(mem) and _is_marker(mem[k]):
                    drop.add(k)
                    k += 1
            elif key:
                seen.append(key)
        kept = [m for i, m in enumerate(mem) if i not in drop]
        out: list = []
        for idx, m in enumerate(kept):
            role = m.get("role")
            if _is_time_line(m):
                nxt = kept[idx + 1] if idx + 1 < len(kept) else None
                if nxt is None or nxt.get("role") != "user":
                    continue  # its question is gone
            elif _is_marker(m):
                if not (out and out[-1].get("role") == "assistant"):
                    continue  # marks no reply
            elif role == "assistant":
                prev = out[-1] if out else None
                # A reply whose question is gone. (A summary note — a system
                # message that is neither clock nor marker — may precede the
                # first reply kept after a compaction.)
                if prev is None or not (
                    prev.get("role") == "user"
                    or (prev.get("role") == "system" and not _is_time_line(prev) and not _is_marker(prev))
                ):
                    continue
            elif role == "user" and isinstance(m.get("content"), str):
                # An unanswered question followed by another one: fold them, as
                # _add_message does live, under the newer clock line.
                j = len(out) - 1
                cur_time = out[j] if j >= 0 and _is_time_line(out[j]) else None
                if cur_time is not None:
                    j -= 1
                if j >= 0 and out[j].get("role") == "user" and isinstance(out[j].get("content"), str):
                    prev_user = out[j]
                    prev_time = j >= 1 and _is_time_line(out[j - 1])
                    del out[j:]
                    if cur_time is not None:
                        if prev_time:
                            out[-1] = cur_time
                        else:
                            out.append(cur_time)
                    merged = dict(m)
                    merged["content"] = _merge_user_turns(prev_user["content"], m["content"])
                    out.append(merged)
                    continue
            out.append(m)
        removed = len(mem) - len(out)
        if removed:
            mem[:] = out
            logger.info(f"[memory] tidied: {removed} message(s) removed, {len(out)} kept")
        return removed

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        merged = 0
        deduped = 0
        seen_replies: set[str] = set()
        for msg in messages:
            content = msg["content"]
            if not (isinstance(content, str) and content):
                logger.warning(f"Skipping invalid message from history: {msg}")
                continue
            if msg["role"] == "system":
                # The interrupt marker, replayed as handle_interrupt writes it
                # live. Mapped to "assistant" it read as a reply she had given —
                # "[Interrupted by user]" as her answer, 81 requests on
                # 2026-10-23 — and the user turns around it were merged across
                # it. A run of markers (one per stray barge-in) says no more
                # than one.
                role = "system" if self.interrupt_method == "system" else "user"
                if self._memory and self._memory[-1] == {"role": role, "content": content}:
                    continue
                self._memory.append({"role": role, "content": content})
                continue
            role = "user" if msg["role"] == "human" else "assistant"
            content = unglue(content)
            # Same collapse as the live path: a stored history is full of user
            # turns whose replies were interrupted, and replaying them as
            # separate questions makes the model answer a stale one.
            # Drop contentless assistant turns (bare "..." left by interrupts).
            # Replayed as context they teach the model that an empty reply is a
            # valid answer, and it starts producing them on its own.
            # Clean the tags older replies leaked into storage. Replayed, they are
            # few-shot examples: the model copied "[expression: joy]", which the
            # live path could not parse — the loop that made every sentence
            # neutral. Labelled emotions become the prompt's own [joy] form (the
            # example it SHOULD copy); [g:...] and [emoji: ...] go. The files are
            # left as they are; only what the model sees is cleaned.
            if role == "assistant":
                content = strip_labelled_tags(
                    _GESTURE_TAG.sub("", normalize_labelled_tags(content))
                )
                content = re.sub(r"[ \t]{2,}", " ", content).strip()
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
                    # ...and the question it answered. Left in, that user turn
                    # was "unanswered" and got GLUED to the next one below
                    # ("Yeah, what time is it?\nHow are you doing today?"), and
                    # the context then read as "two questions -> she says it
                    # makes no sense" — which she then kept saying (2026-11-04
                    # 07:59, the "not English" loop).
                    if self._memory and self._memory[-1].get("role") == "user":
                        self._memory.pop()
                    continue
                seen_replies.add(_norm)
            if role == "user" and self._memory and self._memory[-1]["role"] == "user":
                self._memory[-1]["content"] = _merge_user_turns(
                    self._memory[-1]["content"], content
                )
                merged += 1
                continue
            self._memory.append({"role": role, "content": content})
        self.tidy_memory()
        logger.info(
            f"Loaded {len(self._memory)} messages from history"
            + (f" ({merged} unanswered user turns merged)" if merged else "")
            + (f" ({deduped} repeated replies dropped)" if deduped else "")
        )
        self.schedule_prefix_warm()

    def schedule_post_reply_warm(self, delay: float = 0.15) -> None:
        """After a reply is stored: prefill system + memory (now ending with
        that reply) so the NEXT turn's prompt is cached up to its new tail
        (time line + the user's audio). Without it every turn re-prefilled the
        previous reply and its question: 103-128 ms to the first token live vs
        58-86 ms for the same audio on a warm prefix (2026-11-04). Debounced —
        several assistant writes in one turn give one warm. LLM_POST_REPLY_WARM=0
        turns it off."""
        if os.getenv("LLM_POST_REPLY_WARM", "1") == "0":
            return
        import asyncio

        pending = getattr(self, "_post_warm_task", None)
        if pending is not None and not pending.done():
            pending.cancel()

        async def _later() -> None:
            await asyncio.sleep(delay)
            self.schedule_prefix_warm()

        try:
            self._post_warm_task = asyncio.get_running_loop().create_task(_later())
        except RuntimeError:
            pass

    def schedule_prefix_warm(self) -> None:
        """Prefill the LLM's cache with system prompt + memory in the background,
        so the first turn of a session does not pay for it. Called whenever the
        memory is (re)established; the work runs in a task."""
        if not isinstance(self._llm, OpenAICompatibleAsyncLLM):
            return
        tools = (
            self._formatted_tools_openai
            if self._use_mcpp and self._tool_manager
            else None
        )
        try:
            import asyncio

            asyncio.get_running_loop().create_task(
                self._llm.warm_prefix(list(self._memory), self._system, tools)
            )
        except RuntimeError:
            pass  # no loop (sync context) — skip; the first turn just runs cold

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        # The reply was already judged a non-answer / repeat and its turn
        # removed: writing the heard copy back (+ a marker) is exactly how the
        # dropped reply returned to the context as an orphan, 160 times on
        # 2026-11-04. Nothing to record.
        if getattr(self, "_turn_dropped", False):
            logger.info("[memory] interrupted turn was already dropped from context; nothing recorded")
            return
        heard = (heard_response or "").strip()
        if heard:
            earlier = self._memory[:-1] if self._memory and self._memory[-1].get("role") == "assistant" else self._memory
            key = _norm_reply(heard)
            if is_non_answer(heard) or _near_repeat(
                key, [_norm_reply(m.get("content", "")) for m in earlier if m.get("role") == "assistant"]
            ):
                if self._memory and self._memory[-1].get("role") == "assistant":
                    self._memory.pop()
                self._drop_open_turn()
                logger.info(f"[memory] interrupted non-answer/repeat (and its turn) not kept: {heard[:60]!r}")
                return

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

    def replace_user_text(self, old: str, new: str) -> bool:
        """Swap the newest user message equal to `old` (or containing it, after a
        merge) for `new` — the transcript of an audio-first speculative turn."""
        for msg in reversed(self._memory):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
                continue
            if msg["content"] == old:
                msg["content"] = new
                return True
            if old in msg["content"]:
                msg["content"] = msg["content"].replace(old, new)
                return True
        return False

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
        self.tidy_memory()
        # Copies of the dicts too: _add_message below may rewrite the last one.
        messages = [dict(m) for m in self._memory]
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

        # omni mode: the turn's own audio goes to a speech-in LLM. The ASR text
        # stays OUT of the live request (sent too, the model trusts the transcript
        # over its ears — the very error this mode exists to remove; set
        # OMNI_AUDIO_WITH_TEXT=1 to send both) but it IS what memory keeps below,
        # so the history stays text and replays to any provider.
        audio_part = None
        if input_data.audio is not None and getattr(self._llm, "accepts_audio", False):
            audio_part = {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:{input_data.audio.mime_type};base64,{input_data.audio.data}"
                },
            }
            # (the audio-first placeholder is never sent, whatever the switch says)
            if os.getenv("OMNI_AUDIO_WITH_TEXT", "0") != "1" or text_prompt == SPOKEN_PLACEHOLDER:
                user_content = [p for p in user_content if p.get("type") != "text"]
            user_content.append(audio_part)
            heard = getattr(self, "_heard", None)
            if heard:
                user_content.append({"type": "text", "text": f'(What I said, as heard: "{heard}")'})
                # Memory keeps these words too, rather than the audio-first
                # placeholder; Whisper's later transcript still goes to the
                # history file and the chat panel (speculative.set_text finds no
                # placeholder here and leaves this text in place).
                if text_prompt == SPOKEN_PLACEHOLDER:
                    text_prompt = heard
            _dump_omni_turn(input_data.audio.data, text_prompt, self._memory)
            # base64 of 16 kHz 16-bit mono: 4/3 chars per byte, 32 kB per second
            logger.info(
                f"[omni] user turn sent as audio "
                f"(~{len(input_data.audio.data) * 3 / 4 / 32000:.1f}s"
                f"{', + ASR text' if len(user_content) > 1 else ''}); "
                f"ASR text kept for memory: {text_prompt[:60]!r}"
            )

        if user_content:
            # The clock, every turn. Asked the time, the model answered from nothing
            # ("It's 3:17 PM." at 17:31) — the persona does not tell it to use the
            # time tool and it did not. Sent just before the user turn, not in the
            # system prompt: a prompt that changes every minute would void vLLM's
            # prefix cache for the whole history behind it.
            now = datetime.now().astimezone()
            time_message = {
                "role": "system",
                "content": f"Current local time: {now.strftime('%A, %d %B %Y, %H:%M')} "
                f"(UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]}).",
            }

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if (
                not skip_memory
                and text_prompt
                and not input_data.images
                and audio_part is None  # an audio turn is sent as built, never folded
                and self._memory
                and self._memory[-1]["role"] == "user"
                and isinstance(self._memory[-1].get("content"), str)
            ):
                # This turn folds into an unanswered one (see _add_message), so
                # the request IS the memory after the fold: [..., its time line,
                # "A\nB"]. Building it as copy + time + B instead sent B twice —
                # the shallow copy shared the user dict the fold then rewrote
                # ("A\nB", time, "B") — and the next request no longer matched
                # this one as a prefix.
                self._add_message(text_prompt, "user")
                # The unanswered turn may be hours old; its clock line must not be.
                prev = self._memory[-2] if len(self._memory) >= 2 else None
                if (
                    prev is not None
                    and prev.get("role") == "system"
                    and str(prev.get("content", "")).startswith("Current local time:")
                ):
                    self._memory[-2] = dict(time_message)
                return [dict(m) for m in self._memory]

            messages.append(time_message)
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            if not skip_memory:
                # Kept in memory too, so this request is an exact prefix of the
                # next one. Left out of memory, it sat between the history and the
                # new turn and was gone from the next request: the hybrid
                # (Gated DeltaNet) Qwen3.8 caches its state only at block
                # boundaries and at the end of a prompt, so every turn re-read up
                # to a block of history (TTFT 0.54-0.81s against 0.23-0.28s).
                # Skipped when this turn merges into an unanswered user turn, so
                # the merge below still sees a user message last.
                if not (self._memory and self._memory[-1]["role"] == "user"):
                    self._memory.append(dict(time_message))
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
            # The first words are held until they show the reply is not asking
            # back (askback_opening) — a few tokens, ~20-40 ms.
            ask_buf = ""
            ask_open = (
                _ASKBACK_GUARD and tool_rounds == 0 and not getattr(self, "_askback_retried", True)
                and getattr(self, "_turn_words", 0) >= 3
            )
            ask_retry = False
            ask_why = ""
            recent_openings = [
                _norm_words(m.get("content") if isinstance(m.get("content"), str) else "")
                for m in self._memory[-16:] if m.get("role") == "assistant"
            ][-8:] if ask_open else []
            # On the retry: "Are you asking about X? If so, <answer>" loses its
            # question (strip_askback) — 03:32, half the retries opened that way.
            trim_open = (
                _ASKBACK_GUARD and tool_rounds == 0 and getattr(self, "_askback_retried", False)
            )
            trim_buf = ""

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
                        # The model started the reply over (transformers.cut_repeat):
                        # stop generating and keep only the first telling for
                        # memory. The sentence divider keeps the partial repeat
                        # already streamed from being spoken.
                        cut = cut_repeat(current_turn_text)
                        if cut is not None:
                            logger.warning(
                                f"[repeat-guard] reply started over after {len(cut)} "
                                f"chars — generation stopped"
                            )
                            current_turn_text = cut
                            await stream.aclose()
                            break
                        # Suppress RAW "<tool_call>...</tool_call>" leaking as
                        # text (hermes format; happens when tools are withheld
                        # after the round cap) — it must never reach TTS.
                        emit, suppress_buf, in_tool_txt = _filter_tool_call_text(
                            event, suppress_buf, in_tool_txt
                        )
                        if emit and trim_open:
                            trim_buf += emit
                            first_end = re.search(r"[.?!]\s", trim_buf)
                            if first_end is None and len(trim_buf) < 200:
                                continue
                            if first_end is not None and len(re.findall(r"[\w']+", trim_buf[first_end.end():])) < 3:
                                if len(trim_buf) < 300:
                                    continue
                            trim_open = False
                            cut = strip_askback(trim_buf)
                            if cut is not None:
                                logger.info(f"[askback-guard] retry still asked first; spoken without it: {cut[:60]!r}")
                                current_turn_text = cut
                                emit = cut
                            else:
                                emit = trim_buf
                            trim_buf = ""
                        if emit and ask_open:
                            ask_buf += emit
                            v_ask = askback_opening(ask_buf)
                            v_rep = repeat_opening(ask_buf, recent_openings)
                            if v_ask or v_rep:
                                ask_open = False
                                ask_retry, ask_why = True, ("asking back" if v_ask else "repeating an earlier reply")
                                await stream.aclose()
                                break
                            if v_ask is None or v_rep is None:
                                continue
                            ask_open = False
                            emit, ask_buf = ask_buf, ""
                        if emit:
                            yield emit
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        self._turn_used_tool = True
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
            if ask_retry:
                self._askback_retried = True
                logger.info(f"[askback-guard] reply opened by {ask_why}: {ask_buf[:70]!r} — asked again")
                _note = _REPEAT_NUDGE if ask_why.startswith("repeat") else _ASKBACK_NUDGE
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        c = messages[i].get("content")
                        c = (c + " " + _note) if isinstance(c, str) else (
                            list(c or []) + [{"type": "text", "text": _note}]
                        )
                        messages[i] = {**messages[i], "content": c}
                        break
                current_turn_text = ""
                continue
            if ask_buf:
                yield ask_buf  # a reply shorter than the check
                ask_buf = ""
            if trim_buf:
                cut = strip_askback(trim_buf)
                if cut is not None:
                    current_turn_text = cut
                yield cut if cut is not None else trim_buf
                trim_buf = ""
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
                if (
                    tools_for_api
                    and not self._turn_used_tool
                    and not getattr(self, "_nudged", False)
                    and tool_rounds < max_tool_rounds
                    and is_announce_only(current_turn_text)
                ):
                    # Promised a lookup and made none: ask once, go round again.
                    # The promise was already spoken; the next round's answer
                    # follows it. Memory keeps only the final round's text.
                    self._nudged = True
                    logger.info(
                        f"[tool-nudge] promised a lookup without calling a tool: "
                        f"{current_turn_text[:60]!r} — asking for the call"
                    )
                    messages.append({"role": "assistant", "content": current_turn_text})
                    messages.append(
                        {"role": "system", "content": _TOOL_NUDGE.get(self._language, _TOOL_NUDGE["en"])}
                    )
                    continue
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

            # omni: what the Thinker hears in this turn's audio, written out
            # first (openai_omni_llm.transcribe) and sent in beside it.
            self._heard = None
            if input_data.audio is not None and hasattr(self._llm, "transcribe"):
                from ... import omni_addressee

                if omni_addressee.reply_needs_heard(input_data.audio.data):
                    self._heard = await self._llm.transcribe(input_data.audio.data)
            self._turn_words = len(re.findall(r"[\w']+", self._heard or " ".join(
                getattr(t, "content", "") or "" for t in (getattr(input_data, "texts", None) or [])
            )))
            self._askback_retried = False
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
    def _mem_tokens(memory: List[Dict[str, Any]], cal: float = 1.0) -> int:
        """Conservative token estimate (JA/tool-dense ~= chars/1.3), scaled by
        `cal` — the LLM's measured real/estimated ratio (see _token_cal). The
        default 1.0 is the raw estimate."""
        n = 0
        for m in memory:
            c = m.get("content")
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                n += sum(len(str(p.get("text", ""))) for p in c if isinstance(p, dict))
        return int(int(n / 1.3) * _safe_cal(cal))

    def _token_cal(self) -> float:
        """The LLM's calibration factor for the char estimate (AsyncLLM._cal,
        an EMA of real/estimated prompt tokens). 1.0 when the provider keeps
        none or has seen no usage yet — the compaction threshold then reads
        the raw estimate exactly as before."""
        return _safe_cal(getattr(getattr(self, "_llm", None), "_cal", 1.0))

    def schedule_compaction_if_needed(self) -> None:
        """If memory is near the cap, kick off a background summarization. Cheap
        guard so it's safe to call every turn; the real work runs in a task."""
        if self._compacting:
            return
        if self._mem_tokens(self._memory, self._token_cal()) < self._COMPACT_THRESHOLD_TOKENS:
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
                f"~{self._mem_tokens(self._memory, self._token_cal())} tok)"
            )
            # The summary replaced the start of the prompt; re-warm the new one.
            self.schedule_prefix_warm()
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
        # chat_completion reports API failures by YIELDING an error sentence (so
        # a live turn speaks it) rather than raising. Taken as the summary, it
        # replaced 337 messages of context with one error line (2026-10-23).
        if out.lstrip().startswith("Error calling the chat endpoint"):
            logger.warning(f"[compact] summarize LLM call failed: {out.strip()[:120]}")
            return ""
        return out.strip()

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False
        self._turn_dropped = False
        self._turn_used_tool = False
        self._nudged = False

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
