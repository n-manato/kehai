from dataclasses import dataclass, asdict
from typing import List, Optional
from abc import ABC, abstractmethod


@dataclass
class Actions:
    """Represents actions that can be performed alongside text output"""

    expressions: Optional[List[str] | List[int]] = None
    motions: Optional[List[str]] = None
    pictures: Optional[List[str]] = None
    sounds: Optional[List[str]] = None
    # Raw emotion keyword(s) for this sentence (e.g. "joy_grin", "anger_pout"),
    # sent EVERY sentence. The Live2D `expressions`/`motions` collapse these into
    # a fixed model vocabulary; the MMD face maps the fine-grained keyword to a
    # richer expression preset, so it's carried through untouched.
    emotions: Optional[List[str]] = None
    # Bespoke MMD body-motion handle for this sentence: {"id","key","duration"}.
    # The backend fires generation in parallel with TTS (before this payload is
    # sent) and the frontend fetches /motion/clip/{id} at speech start, falling
    # back to the fresh /motion/pool/{key} buffer if it isn't ready. Only set
    # when the motion subsystem is enabled; to_dict() drops it otherwise.
    motion: Optional[dict] = None
    # Content GESTURE intent the LLM tagged this sentence with (e.g. "point",
    # "count", "shrug") — drives a sentence-matching body clip. None -> the body
    # falls back to the emotion.
    gesture: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert Actions object to a dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class BaseOutput(ABC):
    """Base class for agent outputs that can be iterated"""

    @abstractmethod
    def __aiter__(self):
        """Make the output iterable"""
        pass


@dataclass
class DisplayText:
    """Text to be displayed with optional metadata"""

    text: str
    name: Optional[str] = "AI"  # Keep the name field for frontend display
    avatar: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {"text": self.text, "name": self.name, "avatar": self.avatar}

    def __str__(self) -> str:
        """String representation for logging"""
        return f"{self.name}: {self.text}"


@dataclass
class SentenceOutput(BaseOutput):
    """
    Output type for text-based responses.
    Contains a single sentence pair (display and TTS) with associated actions.

    Attributes:
        display_text: Text to be displayed in UI
        tts_text: Text to be sent to TTS engine
        actions: Associated actions (expressions, pictures, sounds)
    """

    display_text: DisplayText  # Changed from str to DisplayText
    tts_text: str  # Text for TTS
    actions: Actions

    async def __aiter__(self):
        """Yield the sentence pair and actions"""
        yield self.display_text, self.tts_text, self.actions


@dataclass
class AudioOutput(BaseOutput):
    """Output type for audio-based responses"""

    audio_path: str
    display_text: DisplayText  # Changed from str to DisplayText
    transcript: str  # Original transcript
    actions: Actions

    async def __aiter__(self):
        """Iterate through audio segments and their actions"""
        yield self.audio_path, self.display_text, self.transcript, self.actions
