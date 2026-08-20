from typing import Type
from .asr_interface import ASRInterface


class ASRFactory:
    @staticmethod
    def get_asr_system(system_name: str, **kwargs) -> Type[ASRInterface]:
        if system_name == "faster_whisper":
            from .faster_whisper_asr import VoiceRecognition as FasterWhisperASR

            return FasterWhisperASR(
                model_path=kwargs.get("model_path"),
                download_root=kwargs.get("download_root"),
                language=kwargs.get("language"),
                device=kwargs.get("device"),
                compute_type=kwargs.get("compute_type"),
                prompt=kwargs.get("prompt", None),
            )
        elif system_name == "whisper_cpp":
            from .whisper_cpp_asr import VoiceRecognition as WhisperCPPASR

            return WhisperCPPASR(**kwargs)
        elif system_name == "whisper":
            from .openai_whisper_asr import VoiceRecognition as WhisperASR

            return WhisperASR(**kwargs)
        elif system_name == "fun_asr":
            from .fun_asr import VoiceRecognition as FunASR

            return FunASR(
                model_name=kwargs.get("model_name"),
                vad_model=kwargs.get("vad_model"),
                punc_model=kwargs.get("punc_model"),
                ncpu=kwargs.get("ncpu"),
                hub=kwargs.get("hub"),
                device=kwargs.get("device"),
                language=kwargs.get("language"),
                use_itn=kwargs.get("use_itn"),
                # sample_rate=kwargs.get("sample_rate"),
            )
        elif system_name == "azure_asr":
            from .azure_asr import VoiceRecognition as AzureASR

            return AzureASR(
                subscription_key=kwargs.get("api_key"),
                region=kwargs.get("region"),
                languages=kwargs.get("languages", ["en-US", "zh-CN"]),
            )
        elif system_name == "groq_whisper_asr":
            from .groq_whisper_asr import VoiceRecognition as GroqWhisperASR

            return GroqWhisperASR(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model"),
                lang=kwargs.get("lang"),
            )
        elif system_name == "sherpa_onnx_asr":
            from .sherpa_onnx_asr import VoiceRecognition as SherpaOnnxASR

            return SherpaOnnxASR(**kwargs)
        elif system_name == "gateway_whisper_asr":
            from .gateway_whisper_asr import VoiceRecognition as GatewayWhisperASR

            # Drop Nones so the engine's defaults win when config omits a field.
            return GatewayWhisperASR(
                **{k: v for k, v in kwargs.items() if v is not None}
            )
        elif system_name == "elevenlabs_asr":
            from .elevenlabs_asr import VoiceRecognition as ElevenLabsASR

            return ElevenLabsASR(**{k: v for k, v in kwargs.items() if v is not None})
        elif system_name == "elevenlabs_realtime_asr":
            from .elevenlabs_realtime_asr import VoiceRecognition as ElevenRealtimeASR

            return ElevenRealtimeASR(
                **{k: v for k, v in kwargs.items() if v is not None}
            )
        elif system_name == "shadow_asr":
            from .shadow_asr import VoiceRecognition as ShadowASR

            # `_parent` carries the whole asr_config so this can build the two
            # engines it compares without their settings being duplicated here.
            return ShadowASR(**{k: v for k, v in kwargs.items() if v is not None})
        else:
            raise ValueError(f"Unknown ASR system: {system_name}")
