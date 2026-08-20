####
# change from xTTS.py
####

import re
import time
import requests
from loguru import logger
from .tts_interface import TTSInterface

# How long to wait for the GPT-SoVITS server before giving up on a sentence.
# Synthesis is serialized (see _tts_synthesis_lock in tts_manager), so this
# timeout covers ONE sentence's own synthesis — not time queued behind others.
# On GPU a sentence is ~1-7s (proportional to length). Occasional spikes to
# ~30s happen when the GPU is contended (e.g. the browser rendering Live2D +
# lip-sync while earlier audio plays). 60s gives ~2x headroom over the worst
# observed spike while still bailing on a genuinely hung request. (Cold-start on
# the first sentence is handled separately by warming GPT-SoVITS at launch.)
_REQUEST_TIMEOUT = 60


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        aux_ref_audio_paths: list[str] | None = None,
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        split_bucket: bool = True,
        parallel_infer: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        media_type: str = "wav",
        streaming_mode: str = "false",
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.aux_ref_audio_paths = aux_ref_audio_paths or []
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.split_bucket = split_bucket
        self.parallel_infer = parallel_infer
        self.speed_factor = speed_factor
        self.fragment_interval = fragment_interval
        self.media_type = media_type
        self.streaming_mode = streaming_mode

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)
        # Prepare the data for the POST request
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "split_bucket": self.split_bucket,
            "parallel_infer": self.parallel_infer,
            "speed_factor": self.speed_factor,
            "fragment_interval": self.fragment_interval,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
        }
        if self.aux_ref_audio_paths:
            data["aux_ref_audio_paths"] = self.aux_ref_audio_paths

        # Send request to the TTS API. Any failure (timeout, connection drop,
        # non-200) returns None so the caller falls back to a silent payload and
        # the conversation continues instead of stalling on this sentence.
        started = time.monotonic()
        try:
            response = requests.get(self.api_url, params=data, timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"TTS request failed after {time.monotonic() - started:.1f}s: {e}"
            )
            return None

        elapsed = time.monotonic() - started
        if response.status_code == 200:
            with open(file_name, "wb") as audio_file:
                audio_file.write(response.content)
            logger.debug(f"TTS generated in {elapsed:.1f}s for '''{cleaned_text}'''")
            return file_name
        else:
            logger.critical(
                f"Error: Failed to generate audio (HTTP {response.status_code}, "
                f"{elapsed:.1f}s)"
            )
            return None
