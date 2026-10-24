import os
import json
import asyncio
from typing import Callable
from loguru import logger
from fastapi import WebSocket

from prompts import prompt_loader
from .live2d_model import Live2dModel
from .asr.asr_interface import ASRInterface
from .tts.tts_interface import TTSInterface
from .vad.vad_interface import VADInterface
from .agent.agents.agent_interface import AgentInterface
from .translate.translate_interface import TranslateInterface

from .mcpp.server_registry import ServerRegistry
from .mcpp.tool_manager import ToolManager
from .mcpp.mcp_client import MCPClient
from .mcpp.tool_executor import ToolExecutor
from .mcpp.tool_adapter import ToolAdapter

from .asr.asr_factory import ASRFactory
from .tts.tts_factory import TTSFactory
from .vad.vad_factory import VADFactory
from .agent.agent_factory import AgentFactory
from .translate.translate_factory import TranslateFactory

from .config_manager import (
    Config,
    AgentConfig,
    CharacterConfig,
    SystemConfig,
    ASRConfig,
    TTSConfig,
    VADConfig,
    TranslatorConfig,
    TTSPreprocessorConfig,
    read_yaml,
    validate_config,
)


# Weights currently loaded into the SHARED PC-B GSV server (process-wide: one
# GSV instance serves every connection, so this is global state, not per-context).
# None = unknown (whatever tts_infer.yaml loaded at GSV boot — the Rem weights).
_GSV_WEIGHTS_STATE: dict = {"applied": None}
# Serializes the swap so two connections (or a connect racing a switch) cannot both
# POST it, and so ensure_gsv_weights() can safely be awaited from the TTS path.
_GSV_WEIGHTS_LOCK = asyncio.Lock()

# Filled into live2d_expression_prompt.txt at [<insert_punctuation_rule>]. The
# prompt is shared by every character, so this cannot be written once for one
# language: an English persona given the Japanese rule is told to end sentences
# with 。！？ and shown 「うれしいです。😊」 as the model to copy.
_PUNCTUATION_RULE = {
    "ja": (
        "文の終わりには必ず句読点（。！？など）を打ち、絵文字はその後に置いてください。"
        "例：「うれしいです。😊」（「うれしいです😊」は不可）。"
    ),
    "en": (
        "Always end every sentence with a full stop, question mark or exclamation "
        'mark BEFORE any emoji — e.g. write "Glad to hear it. 😊", not '
        '"Glad to hear it 😊".'
    ),
    "zh": (
        "每句话都必须先以句号、问号或感叹号结尾，表情符号放在标点之后，"
        "例如“很高兴听到这个。😊”，而不是“很高兴听到这个😊”。"
    ),
}


class ServiceContext:
    """Initializes, stores, and updates the asr, tts, and llm instances and other
    configurations for a connected client."""

    def __init__(self):
        self.config: Config = None
        self.system_config: SystemConfig = None
        self.character_config: CharacterConfig = None

        self.live2d_model: Live2dModel = None
        self.asr_engine: ASRInterface = None
        self.tts_engine: TTSInterface = None
        self.agent_engine: AgentInterface = None
        # translate_engine can be none if translation is disabled
        self.vad_engine: VADInterface | None = None
        self.translate_engine: TranslateInterface | None = None

        self.mcp_server_registery: ServerRegistry | None = None
        self.tool_adapter: ToolAdapter | None = None
        self.tool_manager: ToolManager | None = None
        self.mcp_client: MCPClient | None = None
        self.tool_executor: ToolExecutor | None = None

        # the system prompt is a combination of the persona prompt and live2d expression prompt
        self.system_prompt: str = None

        # Store the generated MCP prompt string (if MCP enabled)
        self.mcp_prompt: str = ""

        self.history_uid: str = ""  # Add history_uid field

        self.send_text: Callable = None
        self.client_uid: str = None

    def __str__(self):
        return (
            f"ServiceContext:\n"
            f"  System Config: {'Loaded' if self.system_config else 'Not Loaded'}\n"
            f"    Details: {json.dumps(self.system_config.model_dump(), indent=6) if self.system_config else 'None'}\n"
            f"  Live2D Model: {self.live2d_model.model_info if self.live2d_model else 'Not Loaded'}\n"
            f"  ASR Engine: {type(self.asr_engine).__name__ if self.asr_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.asr_config.model_dump(), indent=6) if self.character_config.asr_config else 'None'}\n"
            f"  TTS Engine: {type(self.tts_engine).__name__ if self.tts_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.tts_config.model_dump(), indent=6) if self.character_config.tts_config else 'None'}\n"
            f"  LLM Engine: {type(self.agent_engine).__name__ if self.agent_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.agent_config.model_dump(), indent=6) if self.character_config.agent_config else 'None'}\n"
            f"  VAD Engine: {type(self.vad_engine).__name__ if self.vad_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.vad_config.model_dump(), indent=6) if self.character_config.vad_config else 'None'}\n"
            f"  System Prompt: {self.system_prompt or 'Not Set'}\n"
            f"  MCP Enabled: {'Yes' if self.mcp_client else 'No'}"
        )

    # ==== Initializers

    async def _init_mcp_components(self, use_mcpp, enabled_servers):
        """Initializes MCP components based on configuration, dynamically fetching tool info."""
        logger.debug(
            f"Initializing MCP components: use_mcpp={use_mcpp}, enabled_servers={enabled_servers}"
        )

        # Reset MCP components first
        self.mcp_server_registery = None
        self.tool_manager = None
        self.mcp_client = None
        self.tool_executor = None
        self.json_detector = None
        self.mcp_prompt = ""

        if use_mcpp and enabled_servers:
            # 1. Initialize ServerRegistry
            self.mcp_server_registery = ServerRegistry()
            logger.info("ServerRegistry initialized or referenced.")

            # 2. Use ToolAdapter to get the MCP prompt and tools
            if not self.tool_adapter:
                logger.error(
                    "ToolAdapter not initialized before calling _init_mcp_components."
                )
                self.mcp_prompt = "[Error: ToolAdapter not initialized]"
                return  # Exit if ToolAdapter is mandatory and not initialized

            try:
                (
                    mcp_prompt_string,
                    openai_tools,
                    claude_tools,
                ) = await self.tool_adapter.get_tools(enabled_servers)
                # Store the generated prompt string
                self.mcp_prompt = mcp_prompt_string
                logger.info(
                    f"Dynamically generated MCP prompt string (length: {len(self.mcp_prompt)})."
                )
                logger.info(
                    f"Dynamically formatted tools - OpenAI: {len(openai_tools)}, Claude: {len(claude_tools)}."
                )

                # 3. Initialize ToolManager with the fetched formatted tools

                _, raw_tools_dict = await self.tool_adapter.get_server_and_tool_info(
                    enabled_servers
                )
                self.tool_manager = ToolManager(
                    formatted_tools_openai=openai_tools,
                    formatted_tools_claude=claude_tools,
                    initial_tools_dict=raw_tools_dict,
                )
                logger.info("ToolManager initialized with dynamically fetched tools.")

            except Exception as e:
                logger.error(
                    f"Failed during dynamic MCP tool construction: {e}", exc_info=True
                )
                # Ensure dependent components are not created if construction fails
                self.tool_manager = None
                self.mcp_prompt = "[Error constructing MCP tools/prompt]"

            # 4. Initialize MCPClient
            if self.mcp_server_registery:
                self.mcp_client = MCPClient(
                    self.mcp_server_registery, self.send_text, self.client_uid
                )
                logger.info("MCPClient initialized for this session.")
            else:
                logger.error(
                    "MCP enabled but ServerRegistry not available. MCPClient not created."
                )
                self.mcp_client = None  # Ensure it's None

            # 5. Initialize ToolExecutor
            if self.mcp_client and self.tool_manager:
                self.tool_executor = ToolExecutor(self.mcp_client, self.tool_manager)
                logger.info("ToolExecutor initialized for this session.")
            else:
                logger.warning(
                    "MCPClient or ToolManager not available. ToolExecutor not created."
                )
                self.tool_executor = None  # Ensure it's None

            logger.info("StreamJSONDetector initialized for this session.")

        elif use_mcpp and not enabled_servers:
            logger.warning(
                "use_mcpp is True, but mcp_enabled_servers list is empty. MCP components not initialized."
            )
        else:
            logger.debug(
                "MCP components not initialized (use_mcpp is False or no enabled servers)."
            )

    async def close(self):
        """Clean up resources, especially the MCPClient."""
        logger.info("Closing ServiceContext resources...")
        if self.mcp_client:
            logger.info(f"Closing MCPClient for context instance {id(self)}...")
            await self.mcp_client.aclose()
            self.mcp_client = None
        if self.agent_engine and hasattr(self.agent_engine, "close"):
            await self.agent_engine.close()  # Ensure agent resources are also closed
        logger.info("ServiceContext closed.")

    async def load_cache(
        self,
        config: Config,
        system_config: SystemConfig,
        character_config: CharacterConfig,
        live2d_model: Live2dModel,
        asr_engine: ASRInterface,
        tts_engine: TTSInterface,
        vad_engine: VADInterface,
        agent_engine: AgentInterface,
        translate_engine: TranslateInterface | None,
        mcp_server_registery: ServerRegistry | None = None,
        tool_adapter: ToolAdapter | None = None,
        send_text: Callable = None,
        client_uid: str = None,
    ) -> None:
        """
        Load the ServiceContext with the reference of the provided instances.
        Pass by reference so no reinitialization will be done.
        """
        if not character_config:
            raise ValueError("character_config cannot be None")
        if not system_config:
            raise ValueError("system_config cannot be None")

        self.config = config
        self.system_config = system_config
        self.character_config = character_config
        self.live2d_model = live2d_model
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine
        self.vad_engine = vad_engine
        self.agent_engine = agent_engine
        self.translate_engine = translate_engine
        # Load potentially shared components by reference
        self.mcp_server_registery = mcp_server_registery
        self.tool_adapter = tool_adapter
        self.send_text = send_text
        self.client_uid = client_uid

        # Initialize session-specific MCP components
        await self._init_mcp_components(
            self.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            self.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )

        # Make sure the shared GSV server carries THIS character's fine-tuned
        # weights. Needed here (per client connection) because the boot-time
        # load_from_config runs under run_server.py's temporary asyncio.run()
        # loop — a task created there dies with that loop, so the startup sync
        # never actually executed. Connections run under uvicorn's live loop.
        # Dedup via _GSV_WEIGHTS_STATE makes repeat connections a no-op.
        self._gsv_weights_task = asyncio.create_task(self.sync_gsv_weights())

        logger.debug(f"Loaded service context with cache: {character_config}")

    async def load_from_config(self, config: Config) -> None:
        """
        Load the ServiceContext with the config.
        Reinitialize the instances if the config is different.

        Parameters:
        - config (Dict): The configuration dictionary.
        """
        if not self.config:
            self.config = config

        if not self.system_config:
            self.system_config = config.system_config

        if not self.character_config:
            self.character_config = config.character_config

        # update all sub-configs

        # init live2d from character config
        self.init_live2d(config.character_config.live2d_model_name)

        # init asr from character config
        self.init_asr(config.character_config.asr_config)

        # init tts from character config
        self.init_tts(config.character_config.tts_config)

        # init vad from character config
        self.init_vad(config.character_config.vad_config)

        # Initialize shared ToolAdapter if it doesn't exist yet
        if (
            not self.tool_adapter
            and config.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp
        ):
            if not self.mcp_server_registery:
                logger.info(
                    "Initializing shared ServerRegistry within load_from_config."
                )
                self.mcp_server_registery = ServerRegistry()
            logger.info("Initializing shared ToolAdapter within load_from_config.")
            self.tool_adapter = ToolAdapter(server_registery=self.mcp_server_registery)

        # Initialize MCP Components before initializing Agent
        await self._init_mcp_components(
            config.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            config.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )

        # Propagate the character language into the TTS preprocessor config so
        # the latin->katakana pass only runs for Japanese voices.
        config.character_config.tts_preprocessor_config.language = getattr(
            config.character_config, "language", "ja"
        )

        # init agent from character config
        await self.init_agent(
            config.character_config.agent_config,
            config.character_config.persona_prompt,
            config.character_config.tts_preprocessor_config,
            config.character_config.avatar or "",
        )

        self.init_translate(
            config.character_config.tts_preprocessor_config.translator_config
        )

        # store typed config references
        self.config = config
        self.system_config = config.system_config or self.system_config
        self.character_config = config.character_config

        # Load this character's fine-tuned GSV weights into the shared PC-B
        # server (hot-swap via the gateway). Fire-and-forget: a cold gateway
        # start can take ~60s and must not block startup / config switch.
        self._gsv_weights_task = asyncio.create_task(self.sync_gsv_weights())

    def init_live2d(self, live2d_model_name: str) -> None:
        logger.info(f"Initializing Live2D: {live2d_model_name}")
        try:
            self.live2d_model = Live2dModel(live2d_model_name)
            self.character_config.live2d_model_name = live2d_model_name
        except Exception as e:
            logger.critical(f"Error initializing Live2D: {e}")
            logger.critical("Try to proceed without Live2D...")

    def init_asr(self, asr_config: ASRConfig) -> None:
        if not self.asr_engine or (self.character_config.asr_config != asr_config):
            logger.info(f"Initializing ASR: {asr_config.asr_model}")
            kwargs = getattr(asr_config, asr_config.asr_model).model_dump()
            if asr_config.asr_model == "shadow_asr":
                # The A/B wrapper builds two engines, so it needs their sections
                # too. Passed explicitly rather than widening the factory
                # signature for every engine that does not want it.
                kwargs["_parent"] = asr_config.model_dump()
            self.asr_engine = ASRFactory.get_asr_system(asr_config.asr_model, **kwargs)
            # saving config should be done after successful initialization
            self.character_config.asr_config = asr_config
        else:
            logger.info("ASR already initialized with the same config.")

    def init_tts(self, tts_config: TTSConfig) -> None:
        if not self.tts_engine or (self.character_config.tts_config != tts_config):
            logger.info(f"Initializing TTS: {tts_config.tts_model}")
            self.tts_engine = TTSFactory.get_tts_engine(
                tts_config.tts_model,
                **getattr(tts_config, tts_config.tts_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.tts_config = tts_config
        else:
            logger.info("TTS already initialized with the same config.")

    def init_vad(self, vad_config: VADConfig) -> None:
        if vad_config.vad_model is None:
            logger.info("VAD is disabled.")
            self.vad_engine = None
            return

        if not self.vad_engine or (self.character_config.vad_config != vad_config):
            logger.info(f"Initializing VAD: {vad_config.vad_model}")
            self.vad_engine = VADFactory.get_vad_engine(
                vad_config.vad_model,
                **getattr(vad_config, vad_config.vad_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.vad_config = vad_config
        else:
            logger.info("VAD already initialized with the same config.")

    async def init_agent(
        self,
        agent_config: AgentConfig,
        persona_prompt: str,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        avatar: str = None,
    ) -> None:
        """Initialize or update the LLM engine based on agent configuration.

        tts_preprocessor_config / avatar MUST be passed in from the config being
        loaded. self.character_config is still the PREVIOUS character at this
        point (it is replaced only after every init_* has run, so those can
        compare old vs new), so reading them off self would hand the new agent
        the old character's language -> a Japanese character built with
        language="en" has all of its CJK stripped by speechify and speaks
        nothing at all.
        """
        logger.info(f"Initializing Agent: {agent_config.conversation_agent_choice}")

        if tts_preprocessor_config is None:
            tts_preprocessor_config = self.character_config.tts_preprocessor_config
        if avatar is None:
            avatar = self.character_config.avatar or ""

        if (
            self.agent_engine is not None
            and agent_config == self.character_config.agent_config
            and persona_prompt == self.character_config.persona_prompt
        ):
            # Same brain, but the character around it may still have changed
            # (language, translator, ...) — keep the live agent in sync.
            if getattr(self.agent_engine, "_tts_preprocessor_config", None) is not None:
                self.agent_engine._tts_preprocessor_config = tts_preprocessor_config
                self.agent_engine._language = (
                    getattr(tts_preprocessor_config, "language", "ja") or "ja"
                )
            logger.debug("Agent already initialized with the same config.")
            return

        system_prompt = await self.construct_system_prompt(persona_prompt)

        try:
            self.agent_engine = AgentFactory.create_agent(
                conversation_agent_choice=agent_config.conversation_agent_choice,
                agent_settings=agent_config.agent_settings.model_dump(),
                llm_configs=agent_config.llm_configs.model_dump(),
                system_prompt=system_prompt,
                live2d_model=self.live2d_model,
                tts_preprocessor_config=tts_preprocessor_config,
                character_avatar=avatar,
                system_config=self.system_config.model_dump(),
                tool_manager=self.tool_manager,
                tool_executor=self.tool_executor,
                mcp_prompt_string=self.mcp_prompt,
            )

            logger.debug(f"Agent choice: {agent_config.conversation_agent_choice}")
            logger.debug(f"System prompt: {system_prompt}")

            # Save the current configuration
            self.character_config.agent_config = agent_config
            self.system_prompt = system_prompt

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def init_translate(self, translator_config: TranslatorConfig) -> None:
        """Initialize or update the translation engine based on the configuration."""

        if not translator_config.translate_audio:
            logger.debug("Translation is disabled.")
            return

        if (
            not self.translate_engine
            or self.character_config.tts_preprocessor_config.translator_config
            != translator_config
        ):
            logger.info(
                f"Initializing Translator: {translator_config.translate_provider}"
            )
            self.translate_engine = TranslateFactory.get_translator(
                translator_config.translate_provider,
                getattr(
                    translator_config, translator_config.translate_provider
                ).model_dump(),
            )
            self.character_config.tts_preprocessor_config.translator_config = (
                translator_config
            )
        else:
            logger.info("Translation already initialized with the same config.")

    # ==== utils

    def gsv_stream_config(self):
        """Fresh runtime profile for the streaming TTS path.

        Returns whichever backend THIS character is configured for: an
        ElevenStreamConfig when the character sets tts_config.elevenlabs_stream
        (with a voice_id), otherwise the GSV profile. tts_manager dispatches on
        the returned type, so a character can use a cloned ElevenLabs voice while
        another keeps the fine-tuned GPT-SoVITS one — which is the whole point,
        since Miku's voice is a fine-tune that does not exist on ElevenLabs.

        A new instance every call — tts_manager mutates base_url.
        """
        from .tts.gsv_stream import GSVStreamConfig

        el = (
            getattr(self.character_config.tts_config, "elevenlabs_stream", None)
            if self.character_config
            else None
        )
        if el is not None and getattr(el, "voice_id", ""):
            from .tts.elevenlabs_stream import ElevenStreamConfig

            lang = el.language_code or (
                getattr(self.character_config, "language", "") or "en"
            )
            # Copied BY FIELD NAME rather than as a hand-written argument list.
            # The list version is how `pronunciation_dictionary` would have been
            # added to the YAML schema, documented, and then silently never
            # reached the request — the same shape of miss as the `text_lang` one.
            # Now the two just have to agree on a name.
            #
            # None values are passed through deliberately: for the voice-shaping
            # fields None means "use the settings saved on the voice in
            # ElevenLabs", which is not the same as the dataclass default.
            shared = {
                name: getattr(el, name)
                for name in ElevenStreamConfig.__dataclass_fields__
                if hasattr(el, name)
            }
            # text_lang is derived, not configured: it has no counterpart on the
            # YAML model, so it is set after the copy.
            shared["text_lang"] = lang
            return ElevenStreamConfig(**shared)

        cfg = GSVStreamConfig()
        prof = (
            getattr(self.character_config.tts_config, "gsv_stream", None)
            if self.character_config
            else None
        )
        if prof is not None:
            if prof.ref_audio_path:
                cfg.ref_audio_path = prof.ref_audio_path
            if prof.prompt_text:
                cfg.prompt_text = prof.prompt_text
            if prof.text_lang:
                cfg.text_lang = prof.text_lang
            if prof.prompt_lang:
                cfg.prompt_lang = prof.prompt_lang
        return cfg

    async def sync_gsv_weights(self) -> None:
        """Hot-swap the shared GSV server's fine-tuned weights to this
        character's (gateway /v1/audio/weights -> /set_gpt_weights +
        /set_sovits_weights). No-op when the profile names no weights or the
        wanted weights are already loaded. On failure the voice keeps the
        previously loaded weights — logged loudly so it isn't silent."""
        prof = (
            getattr(self.character_config.tts_config, "gsv_stream", None)
            if self.character_config
            else None
        )
        if prof is None or not (prof.gpt_weights or prof.sovits_weights):
            return
        want = (prof.gpt_weights or "", prof.sovits_weights or "")
        async with _GSV_WEIGHTS_LOCK:
            if _GSV_WEIGHTS_STATE["applied"] == want:
                return
            await self._post_gsv_weights(prof, want)

    async def ensure_gsv_weights(self, timeout: float = 25.0) -> None:
        """Block until the shared GSV server carries THIS character's fine-tune.

        gsv_stream_config() only carries the reference clip, prompt and languages
        per request — the WEIGHTS live on the shared server and are swapped here.
        Firing that swap and forgetting it was silently unreliable: the boot-time
        one dies with run_server.py's temporary loop, and a connect/switch task can
        be cancelled or outrun by the reply, in which case the character speaks with
        the PREVIOUS character's fine-tune (Rem's reference clip through the English
        model — "レムじゃなくなった", while English sounded fine because its weights
        happened to already be loaded). Awaiting it on the TTS path removes every one
        of those races; it is deduped, so it costs nothing from the second turn on.

        Bounded: on timeout the swap keeps running (shielded) and we let the reply
        through rather than stall it behind a cold gateway.
        """
        task = self._gsv_weights_task
        if task is None or task.done():
            task = self._gsv_weights_task = asyncio.create_task(self.sync_gsv_weights())
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"[gsv-weights] still swapping after {timeout:.0f}s — the first lines "
                "may still be in the previous character's voice"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gsv-weights] swap failed: {e!r}")

    async def _post_gsv_weights(self, prof, want) -> None:
        import httpx

        stream_url = os.getenv(
            "GSV_STREAM_URL", "https://gpu.aiccore-uno.ai/v1/audio/stream"
        )
        urls = [stream_url.replace("/audio/stream", "/audio/weights")]
        public = "https://gpu.aiccore-uno.ai/v1/audio/weights"
        if public not in urls:
            urls.append(public)
        key = os.getenv("MOTION_GATEWAY_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        params = {}
        if prof.gpt_weights:
            params["gpt_weights"] = prof.gpt_weights
        if prof.sovits_weights:
            params["sovits_weights"] = prof.sovits_weights

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=10.0)
        ) as client:
            for url in urls:
                try:
                    r = await client.get(url, params=params, headers=headers)
                    if r.status_code == 200:
                        _GSV_WEIGHTS_STATE["applied"] = want
                        logger.info(f"[gsv-weights] loaded {want} via {url}")
                        return
                    logger.warning(
                        f"[gsv-weights] {r.status_code} @ {url}: {r.text[:200]}"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[gsv-weights] {url} failed: {e!r}")
        logger.error(
            f"[gsv-weights] FAILED to load {want} — the voice may still be the "
            "previous character's fine-tune!"
        )

    async def construct_system_prompt(self, persona_prompt: str) -> str:
        """
        Append tool prompts to persona prompt.

        Parameters:
        - persona_prompt (str): The persona prompt.

        Returns:
        - str: The system prompt with all tool prompts appended.
        """
        logger.debug(f"constructing persona_prompt: '''{persona_prompt}'''")

        for prompt_name, prompt_file in self.system_config.tool_prompts.items():
            if (
                prompt_name == "group_conversation_prompt"
                or prompt_name == "proactive_speak_prompt"
            ):
                continue

            prompt_content = prompt_loader.load_util(prompt_file)

            if prompt_name == "live2d_expression_prompt":
                prompt_content = prompt_content.replace(
                    "[<insert_emomap_keys>]", self.live2d_model.emo_str
                )
                # The punctuation rule is language-specific, and this file is
                # shared by every character. It used to hard-code the Japanese
                # case, so the English personas were being told to end sentences
                # with 。！？ and shown a Japanese example.
                lang = (getattr(self.character_config, "language", "") or "ja").lower()
                prompt_content = prompt_content.replace(
                    "[<insert_punctuation_rule>]",
                    _PUNCTUATION_RULE.get(lang, _PUNCTUATION_RULE["en"]),
                )

            if prompt_name == "mcp_prompt":
                continue

            persona_prompt += prompt_content

        # SPEECH-SAFE OUTPUT, per character language.
        # ja: the GSV voice model skips latin text almost entirely (an
        # English-heavy sentence measured 0.5s of audio for ~8s of content —
        # "字幕には出るのに調べた内容をしゃべらない") -> katakana rule.
        # en: plain conversational-speech rules instead (katakana would be wrong).
        lang = (
            getattr(self.character_config, "language", "ja")
            if self.character_config
            else "ja"
        )
        if lang == "ja":
            persona_prompt += (
                "\n\n[音声出力の規則] あなたの返答はそのまま日本語音声合成で読み上げ"
                "られます。英語などの外国語の単語・固有名詞・略語は必ずカタカナ表記"
                "で書いてください(例: University of Nebraska Omaha → ユニバーシティ・"
                "オブ・ネブラスカ・オマハ、AI → エーアイ、URL は読まない)。箇条書き"
                "記号や URL や括弧書きの英字は使わず、話し言葉の文章にしてください。"
            )
        else:
            persona_prompt += (
                "\n\n[Speech output rules] Your reply is read aloud verbatim by an "
                "English text-to-speech voice. Respond ONLY in natural spoken "
                "English: no markdown, no bullet points, no URLs, no code, no "
                "parenthetical asides. Prefer words over symbols (say 'percent', "
                "'about three point five'). Keep sentences short enough to be "
                "spoken in one breath."
            )

        # Every word is spoken, so tool use must be silent and reasoning must stay
        # internal. Qwen3.8 (thinking off) opened tool turns with "Accessing the
        # relevant records now." / "Hmm, let me check." and once wrote its plan
        # into the reply ("The user is asking for the current time. I need to use
        # the get_current_time function...") — all of which went to TTS.
        if lang == "ja":
            persona_prompt += (
                "\n\n[ツールと考えの扱い] ツールが必要なときは、前置きを言わずにすぐ"
                "呼び出してください（「確認しますね」「少々お待ちください」などは"
                "言わない）。考えている過程や手順は書かず、相手への返事だけを"
                "話してください。"
            )
        else:
            persona_prompt += (
                "\n\n[Tools and reasoning] When you need a tool, call it at once "
                "and say nothing before it: no 'let me check', 'one moment', "
                "'retrieving', 'accessing'. Never write out your reasoning, plans "
                "or what the user is asking for; say only your reply to them."
            )

        logger.debug("\n === System Prompt ===")
        logger.debug(persona_prompt)

        return persona_prompt

    async def handle_config_switch(
        self,
        websocket: WebSocket,
        config_file_name: str,
    ) -> None:
        """
        Handle the configuration switch request.
        Change the configuration to a new config and notify the client.

        Parameters:
        - websocket (WebSocket): The WebSocket connection.
        - config_file_name (str): The name of the configuration file.
        """
        try:
            new_character_config_data = None

            if config_file_name == "conf.yaml":
                # Load base config
                new_character_config_data = read_yaml("conf.yaml").get(
                    "character_config"
                )
            else:
                # Load alternative config and merge with base config
                characters_dir = self.system_config.config_alts_dir
                file_path = os.path.normpath(
                    os.path.join(characters_dir, config_file_name)
                )
                if not file_path.startswith(characters_dir):
                    raise ValueError("Invalid configuration file path")

                alt_config_data = read_yaml(file_path).get("character_config")

                # Merge over the BASE conf.yaml character config, NOT the
                # currently active character: alt files are partial overrides
                # of the base (their comments say "inherit from conf.yaml").
                # Merging over the previous character leaked its persona /
                # language / voice profile into presets that don't set them
                # (e.g. EDI-english -> a Rem outfit preset kept English).
                base_character_config = read_yaml("conf.yaml").get("character_config")
                new_character_config_data = deep_merge(
                    base_character_config, alt_config_data
                )

                # The merge carries over the PREVIOUS character's RESOLVED
                # persona_prompt, and the validator only loads
                # persona_prompt_file when persona_prompt is empty — so an alt
                # config that names a persona FILE would silently keep the old
                # persona. Blank it so the new file actually loads.
                if (
                    alt_config_data
                    and alt_config_data.get("persona_prompt_file")
                    and not alt_config_data.get("persona_prompt")
                ):
                    new_character_config_data["persona_prompt"] = ""

            if new_character_config_data:
                new_config = {
                    "system_config": self.system_config.model_dump(),
                    "character_config": new_character_config_data,
                }
                new_config = validate_config(new_config)
                await self.load_from_config(new_config)  # Await the async load
                logger.debug(f"New config: {self}")
                logger.debug(
                    f"New character config: {self.character_config.model_dump()}"
                )

                # Send responses to client
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "set-model-and-conf",
                            "model_info": self.live2d_model.model_info,
                            "conf_name": self.character_config.conf_name,
                            "conf_uid": self.character_config.conf_uid,
                        }
                    )
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "config-switched",
                            "message": f"Switched to config: {config_file_name}",
                        }
                    )
                )

                logger.info(f"Configuration switched to {config_file_name}")
            else:
                raise ValueError(
                    f"Failed to load configuration from {config_file_name}"
                )

        except Exception as e:
            logger.error(f"Error switching configuration: {e}")
            logger.debug(self)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error switching configuration: {str(e)}",
                    }
                )
            )
            raise e


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
