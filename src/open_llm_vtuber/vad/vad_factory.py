from typing import Type
from .vad_interface import VADInterface


class VADFactory:
    @staticmethod
    def get_vad_engine(engine_type, **kwargs) -> Type[VADInterface]:
        if engine_type is None:
            return None
        if engine_type == "silero_vad":
            from .silero import VADEngine as SileroVADEngine

            # Keyword args: the old positional call silently DROPPED every field
            # added to the config afterwards (onset_db_threshold arrived set to
            # its default 0, so the proximity gate looked implemented but was
            # never armed).
            return SileroVADEngine(
                orig_sr=kwargs.get("orig_sr"),
                target_sr=kwargs.get("target_sr"),
                prob_threshold=kwargs.get("prob_threshold"),
                db_threshold=kwargs.get("db_threshold"),
                required_hits=kwargs.get("required_hits"),
                required_misses=kwargs.get("required_misses"),
                smoothing_window=kwargs.get("smoothing_window"),
                end_misses=kwargs.get("end_misses", 12),
                onset_db_threshold=kwargs.get("onset_db_threshold", 0),
                segment_mean_db=kwargs.get("segment_mean_db", 0),
            )
