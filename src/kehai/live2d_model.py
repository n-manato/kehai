import re
import json
import chardet
from loguru import logger

# Broad emoji / pictograph ranges. Used to scrub ALL emojis from subtitle text
# (not only the ones mapped to emotions) so stray emojis the LLM emits don't
# leak into the on-screen text. TTS already strips them separately.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji extensions
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero-width joiner
    "\U000020e3"  # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)

# This class will only prepare the payload for the live2d model
# the process of sending the payload should be done by the caller
# This class is **Not responsible** for sending the payload to the server


class Live2dModel:
    """
    A class to represent a Live2D model. This class only prepares and stores the information of the Live2D model. It does not send anything to the frontend or server or anything.

    Attributes:
        model_dict_path (str): The path to the model dictionary file.
        live2d_model_name (str): The name of the Live2D model.
        model_info (dict): The information of the Live2D model.
        emo_map (dict): The emotion map of the Live2D model.
        emo_str (str): The string representation of the emotion map of the Live2D model.
    """

    model_dict_path: str
    live2d_model_name: str
    model_info: dict
    emo_map: dict
    emo_str: str

    # Maps common emojis to emotion keywords. LLMs naturally sprinkle emojis
    # into their replies, so treating them as emotion triggers gives the Live2D
    # body far more motion variety without forcing the model to emit explicit
    # [tag] markers. Each emoji resolves to a generic emotion keyword; the
    # model's emotionMap / motionGroupMap then refines it to a concrete motion.
    # Only emojis whose emotion exists in the model's emotionMap are used (the
    # rest are ignored at runtime), so this list can stay broad.
    EMOJI_EMOTION_MAP: dict = {
        # joy / happiness
        "😊": "joy_smile", "🙂": "joy_smile", "😀": "joy", "😃": "joy",
        "😄": "joy_big", "😁": "joy_grin", "😆": "joy_big", "🥰": "joy_smile",
        "😍": "joy_smile", "☺️": "joy_smile", "😋": "joy", "🤗": "joy",
        "🎉": "joy_big", "✨": "joy", "💕": "joy_smile", "❤️": "joy_smile",
        "♥️": "joy_smile", "💖": "joy_smile", "👍": "joy", "🌟": "joy",
        "😺": "joy", "😸": "joy_big",
        # shy
        "😳": "shy", "😅": "shy",
        # affection / playfulness — heart-eyes and a wink are their own faces on the VRM
        # (はぁと pupils / ウィンク２). Only the EXPLICITLY romantic emoji map to `love`:
        # ❤️🥰😍💖 are everyday warmth and, with the prompt asking for an emoji on nearly
        # every sentence, routing those here put heart pupils on a third of all lines.
        "😘": "love", "😻": "love", "💘": "love",
        "😉": "wink", "😜": "wink", "😝": "wink",
        # anger
        "😠": "anger", "😡": "anger_rage", "💢": "anger_rage", "😤": "anger_pout",
        "🤬": "anger_rage", "😾": "anger", "👿": "anger_rage",
        # sadness
        "😢": "sadness", "😭": "sadness_tearful", "😞": "sadness_sad",
        "😔": "sadness_sad", "😟": "sadness_worry", "🥺": "sadness_worry",
        "😿": "sadness", "💧": "sadness", "😩": "sadness_sad", "😖": "sadness_sad",
        # surprise / fear
        "😮": "surprise", "😲": "surprise_shock", "😱": "surprise_shock",
        "😨": "fear", "😯": "surprise", "🙀": "surprise_amazed", "‼️": "surprise",
        "❗": "surprise", "⁉️": "surprise",
        # thinking
        "🤔": "thinking", "💭": "thinking", "❓": "thinking",
        # suspicion / doubt — a sideways "Look" glance (🧐 scrutinizing,
        # 🤨 raised-eyebrow skepticism, 👀 side-eye)
        "🧐": "look", "🤨": "look", "👀": "look",
    }

    def __init__(
        self, live2d_model_name: str, model_dict_path: str = "model_dict.json"
    ):
        self.model_dict_path: str = model_dict_path
        self.live2d_model_name: str = live2d_model_name
        self.set_model(live2d_model_name)

    def set_model(self, model_name: str) -> None:
        """
        Set the model with its name and load the model information. This method will initialize the `self.model_info`, `self.emo_map`, and `self.emo_str` attributes.
        This method is called in the constructor.

        Parameters:
            model_name (str): The name of the live2d model.

            Returns:
            None
        """

        self.model_info: dict = self._lookup_model_info(model_name)
        self.emo_map: dict = {
            k.lower(): v for k, v in self.model_info["emotionMap"].items()
        }
        self.emo_str: str = " ".join([f"[{key}]," for key in self.emo_map.keys()])
        # emo_str is a string of the keys in the emoMap dictionary. The keys are enclosed in square brackets.
        # example: `"[fear], [anger], [disgust], [sadness], [joy], [neutral], [surprise]"`

    def _load_file_content(self, file_path: str) -> str:
        """Load the content of a file with robust encoding handling."""
        # Try common encodings first
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        # If all common encodings fail, try to detect encoding
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
            detected = chardet.detect(raw_data)
            detected_encoding = detected["encoding"]

            if detected_encoding:
                try:
                    return raw_data.decode(detected_encoding)
                except UnicodeDecodeError:
                    pass
        except Exception as e:
            logger.error(f"Error detecting encoding for {file_path}: {e}")

        raise UnicodeError(f"Failed to decode {file_path} with any encoding")

    def _lookup_model_info(self, model_name: str) -> dict:
        """
        Find the model information from the model dictionary and return the information about the matched model.

        Parameters:
            model_name (str): The name of the live2d model.

        Returns:
            dict: The dictionary with the information of the matched model.

        Raises:
            FileNotFoundError if the model dictionary file is not found.

            json.JSONDecodeError if the model dictionary file is not a valid JSON file.

            KeyError if the model name is not found in the model dictionary.

        """

        self.live2d_model_name = model_name

        try:
            file_content = self._load_file_content(self.model_dict_path)
            model_dict = json.loads(file_content)
        except FileNotFoundError as file_e:
            logger.critical(
                f"Model dictionary file not found at {self.model_dict_path}."
            )
            raise file_e
        except json.JSONDecodeError as json_e:
            logger.critical(
                f"Error decoding JSON from model dictionary file at {self.model_dict_path}."
            )
            raise json_e
        except UnicodeError as uni_e:
            logger.critical(
                f"Error reading model dictionary file at {self.model_dict_path}."
            )
            raise uni_e
        except Exception as e:
            logger.critical(
                f"Error occurred while reading model dictionary file at {self.model_dict_path}."
            )
            raise e

        # Find the model in the model_dict
        matched_model = next(
            (model for model in model_dict if model["name"] == model_name), None
        )

        if matched_model is None:
            logger.critical(f"Unable to find {model_name} in {self.model_dict_path}.")
            raise KeyError(
                f"{model_name} not found in model dictionary {self.model_dict_path}."
            )

        # The feature: "translate model url to full url if it starts with '/' " is no longer implemented here

        logger.info("Model Information Loaded.")

        return matched_model

    def extract_emotion(self, str_to_check: str) -> list:
        """
        Check the input string for any emotion keywords and return a list of values (the expression index) of the emotions found in the string.

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            list: A list of values of the emotions found in the string. An empty list is returned if no emotions are found.
        """

        expression_list = []
        str_to_check = str_to_check.lower()

        i = 0
        while i < len(str_to_check):
            if str_to_check[i] != "[":
                i += 1
                continue
            for key in self.emo_map.keys():
                emo_tag = f"[{key}]"
                if str_to_check[i : i + len(emo_tag)] == emo_tag:
                    expression_list.append(self.emo_map[key])
                    i += len(emo_tag) - 1
                    break
            i += 1
        return expression_list

    def extract_emotion_keywords(self, str_to_check: str) -> list:
        """
        Like ``extract_emotion`` but returns the matched emotion KEYWORDS
        (e.g. ``"joy_big"``) instead of their numeric indices.

        Numeric indices are ambiguous when several emotions share the same
        value in ``emotionMap`` (e.g. ``joy_big`` and ``surprise_amazed`` may
        both map to ``10``). Keeping the keyword preserves the exact emotion so
        downstream expression/motion resolution stays correct.

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            list: The emotion keywords found, in order. Empty if none.
        """

        keyword_list = []
        str_to_check = str_to_check.lower()

        i = 0
        while i < len(str_to_check):
            if str_to_check[i] != "[":
                i += 1
                continue
            for key in self.emo_map.keys():
                emo_tag = f"[{key}]"
                if str_to_check[i : i + len(emo_tag)] == emo_tag:
                    keyword_list.append(key)
                    i += len(emo_tag) - 1
                    break
            i += 1
        return keyword_list

    def extract_emoji_emotions(self, str_to_check: str) -> list:
        """
        Scan the input string for known emojis and return the emotion keywords
        they map to, in order of appearance. Only emotions that actually exist
        in this model's ``emo_map`` are returned, so emojis whose emotion the
        model can't express are silently ignored.

        Parameters:
            str_to_check (str): The string to scan for emojis.

        Returns:
            list: Emotion keywords (e.g. ``"joy_smile"``) for the emojis found.
        """

        # Match by SUBSTRING, not character-by-character. Many emojis are made of
        # several codepoints (e.g. "❤️" = U+2764 U+FE0F, "☺️", "♥️", "‼️",
        # "⁉️"); a per-character scan only ever sees the individual codepoints and
        # never matches these combined keys, so those emojis would silently
        # contribute no emotion. Scan left-to-right, trying longer keys first so a
        # base+variation-selector emoji isn't shadowed by its bare base codepoint.
        keyword_list = []
        keys = sorted(self.EMOJI_EMOTION_MAP.keys(), key=len, reverse=True)
        i = 0
        while i < len(str_to_check):
            for emoji in keys:
                if str_to_check.startswith(emoji, i):
                    emotion = self.EMOJI_EMOTION_MAP[emoji].lower()
                    if emotion in self.emo_map:
                        keyword_list.append(emotion)
                    i += len(emoji)
                    break
            else:
                i += 1
        return keyword_list

    def remove_emojis(self, target_str: str) -> str:
        """
        Remove ALL emojis / pictographs from the input string (not just the
        ones mapped to emotions). Emojis act purely as motion triggers, so none
        should leak into the subtitle. The mapped emojis are stripped first so
        multi-codepoint ones are handled, then a broad unicode pass removes any
        remaining stray emojis.

        Parameters:
            target_str (str): The string to clean.

        Returns:
            str: The string with emojis removed.
        """

        for emoji in self.EMOJI_EMOTION_MAP.keys():
            if emoji in target_str:
                target_str = target_str.replace(emoji, "")
        return _EMOJI_PATTERN.sub("", target_str)

    def remove_emotion_keywords(self, target_str: str) -> str:
        """
        Remove the emotion keywords from the input string and return the cleaned string.

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            str: The cleaned string with the emotion keywords removed.
        """

        lower_str = target_str.lower()

        for key in self.emo_map.keys():
            lower_key = f"[{key}]".lower()
            while lower_key in lower_str:
                start_index = lower_str.find(lower_key)
                end_index = start_index + len(lower_key)
                target_str = target_str[:start_index] + target_str[end_index:]
                lower_str = lower_str[:start_index] + lower_str[end_index:]
        return target_str

    def resolve_motion_group(self, emotion: str | int | None) -> str:
        """
        Resolve a motion group name from an emotion keyword or emotion index.

        The model dictionary can optionally define `motionGroupMap`. If it does
        not, fall back to a sensible default mapping.
        """

        motion_group_map = self.model_info.get("motionGroupMap", {}) or {}
        default_group = motion_group_map.get("default", "Talk")

        if emotion is None:
            return default_group

        emotion_key = str(emotion).lower()
        if isinstance(emotion, int):
            reverse_map = {v: k for k, v in self.emo_map.items()}
            emotion_key = reverse_map.get(emotion, str(emotion)).lower()

        if emotion_key in motion_group_map:
            return motion_group_map[emotion_key]

        # Fallback heuristic for models whose motionGroupMap doesn't list this
        # emotion. CRUCIAL: only return a guessed group if that group is
        # actually present in this model's motionGroupMap values — different
        # Rem outfits have different motion sets (e.g. some lack "Thinking" or
        # "Surprise_shock"). Guessing a non-existent group would make the
        # frontend try to play a missing motion. When the guess isn't available,
        # fall back to the model's default group.
        heuristic_map = {
            "neutral": "Idle",
            "idle_normal": "Idle",
            "idle_nod": "Idle_nod",
            "idle_sigh": "Idle_sigh",
            "joy": "Joy_big",
            "joy_big": "Joy_big",
            "joy_smile": "Joy_smile",
            "smirk": "Joy_big",
            "anger": "Anger_rage",
            "anger_rage": "Anger_rage",
            "anger_pout": "Anger_pout",
            "anger_serious": "Anger_serious",
            "anger_sulk": "Anger_sulk",
            "disgust": "Sadness_worry",
            "fear": "Surprise_shock",
            "sadness": "Sadness_worry",
            "sadness_worry": "Sadness_worry",
            "sadness_sad": "Sadness_sad",
            "sadness_tearful": "Sadness_tearful",
            "surprise": "Surprise_shock",
            "surprise_shock": "Surprise_shock",
            "surprise_amazed": "Surprise_amazed",
            "thinking": "Thinking",
        }
        guess = heuristic_map.get(emotion_key)
        if guess and guess in motion_group_map.values():
            return guess
        return default_group

    def resolve_motion_groups(self, emotions: list[int] | list[str] | None) -> list[str]:
        """
        Convert a list of emotion values into motion group candidates.
        """

        if not emotions:
            return [self.resolve_motion_group("neutral")]

        groups: list[str] = []
        for emotion in emotions:
            group = self.resolve_motion_group(emotion)
            if group not in groups:
                groups.append(group)
        return groups

    def resolve_expression(self, emotion: str | int | None) -> str:
        """
        Resolve a face expression name from an emotion keyword or emotion index.
        """

        expression_group_map = self.model_info.get("expressionGroupMap", {}) or {}

        if emotion is None:
            return expression_group_map.get("neutral", "face_normal")

        emotion_key = str(emotion).lower()
        if isinstance(emotion, int):
            reverse_map = {v: k for k, v in self.emo_map.items()}
            emotion_key = reverse_map.get(emotion, str(emotion)).lower()

        if emotion_key in expression_group_map:
            return expression_group_map[emotion_key]

        return "face_normal"  # Default expression

    def resolve_expressions(self, emotions: list[int] | list[str] | None) -> list[str]:
        """
        Convert a list of emotion values into expression names.
        """

        if not emotions:
            return [self.resolve_expression("neutral")]

        expressions: list[str] = []
        for emotion in emotions:
            expr = self.resolve_expression(emotion)
            if expr not in expressions:
                expressions.append(expr)
        return expressions
