# config_manager/character.py
from pydantic import Field, field_validator, model_validator
from typing import Dict, ClassVar, Optional
from .i18n import I18nMixin, Description
from .asr import ASRConfig
from .tts import TTSConfig
from .vad import VADConfig
from .tts_preprocessor import TTSPreprocessorConfig

from .agent import AgentConfig


class CharacterConfig(I18nMixin):
    """Character configuration settings."""

    conf_name: str = Field(..., alias="conf_name")
    conf_uid: str = Field(..., alias="conf_uid")
    live2d_model_name: str = Field(..., alias="live2d_model_name")
    character_name: str = Field(default="", alias="character_name")
    # Spoken-language of this character ("ja" | "en"). Single source of truth for
    # every language-dependent stage: ASR language hint, GSV text/prompt lang,
    # latin->katakana preprocessing, endpoint heuristics, prompt speech rules.
    language: str = Field(default="ja", alias="language")
    # Drive the conversational body from GestureLSM ONLY — no KAWAII talk-loops
    # or one-shot emote clips. Those exist because GestureLSM needs ~1.4s of audio
    # before its first clip lands, so something has to move in the meantime; with
    # this on, that gap is covered by the pre-generated GestureLSM BRIDGE instead,
    # which is the same model's motion rather than a different animation style.
    # Per character: the two bodies read very differently, and mixing styles is
    # only jarring on the one whose motion is meant to be generated.
    gesture_lsm_only: bool = Field(default=False, alias="gesture_lsm_only")
    human_name: str = Field(default="Human", alias="human_name")
    avatar: str = Field(default="", alias="avatar")
    persona_prompt: str = Field(default="", alias="persona_prompt")
    persona_prompt_file: Optional[str] = Field(
        default=None, alias="persona_prompt_file"
    )
    agent_config: AgentConfig = Field(..., alias="agent_config")
    asr_config: ASRConfig = Field(..., alias="asr_config")
    tts_config: TTSConfig = Field(..., alias="tts_config")
    vad_config: VADConfig = Field(..., alias="vad_config")
    tts_preprocessor_config: TTSPreprocessorConfig = Field(
        ..., alias="tts_preprocessor_config"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_name": Description(
            en="Name of the character configuration", zh="角色配置名称"
        ),
        "conf_uid": Description(
            en="Unique identifier for the character configuration",
            zh="角色配置唯一标识符",
        ),
        "live2d_model_name": Description(
            en="Name of the Live2D model to use", zh="使用的Live2D模型名称"
        ),
        "character_name": Description(
            en="Name of the AI character in conversation", zh="对话中AI角色的名字"
        ),
        "language": Description(
            en="Spoken language of this character (ja | en)",
            zh="该角色的语音语言（ja | en）",
        ),
        "persona_prompt": Description(
            en="Persona prompt. The persona of your character.", zh="角色人设提示词"
        ),
        "persona_prompt_file": Description(
            en="Path to a text file holding the persona prompt. Used when "
            "persona_prompt is empty; the file content is loaded into persona_prompt.",
            zh="存放人设提示词的文本文件路径。persona_prompt 为空时从该文件加载。",
        ),
        "agent_config": Description(
            en="Configuration for the conversation agent", zh="对话代理配置"
        ),
        "asr_config": Description(
            en="Configuration for Automatic Speech Recognition", zh="语音识别配置"
        ),
        "tts_config": Description(
            en="Configuration for Text-to-Speech", zh="语音合成配置"
        ),
        "vad_config": Description(
            en="Configuration for Voice Activity Detection", zh="语音活动检测配置"
        ),
        "tts_preprocessor_config": Description(
            en="Configuration for Text-to-Speech Preprocessor",
            zh="语音合成预处理器配置",
        ),
        "human_name": Description(
            en="Name of the human user in conversation", zh="对话中人类用户的名字"
        ),
        "avatar": Description(
            en="Avatar image path for the character", zh="角色头像图片路径"
        ),
    }

    @model_validator(mode="after")
    def resolve_persona_prompt(self):
        """Load persona_prompt from persona_prompt_file when not given inline.

        Runs on every validation (startup load and character switch), so any
        code path that reads ``persona_prompt`` transparently gets the resolved
        text. Idempotent: once resolved, the inline value is reused on re-merge.
        """
        if not self.persona_prompt and self.persona_prompt_file:
            # Local import avoids a circular import (utils imports the config models).
            from .utils import load_text_file_with_guess_encoding

            content = load_text_file_with_guess_encoding(self.persona_prompt_file)
            if content is None:
                raise ValueError(
                    f"Could not read persona_prompt_file: {self.persona_prompt_file}"
                )
            self.persona_prompt = content

        if not self.persona_prompt:
            raise ValueError(
                "Persona prompt is empty. Provide 'persona_prompt' inline or "
                "'persona_prompt_file' pointing to a text file."
            )
        return self

    @field_validator("character_name")
    def set_default_character_name(cls, v, values):
        if not v and "conf_name" in values:
            return values["conf_name"]
        return v
