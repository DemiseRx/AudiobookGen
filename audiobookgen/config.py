"""Configuration primitives for AudiobookGen.

This module contains static metadata for voices and default generation
parameters. The values are derived from the KaniTTS project description and can
be extended or customised by the operator without touching the main
application logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class VoiceProfile:
    """Describes a voice preset shipped with the service.

    Attributes
    ----------
    identifier:
        The raw speaker identifier understood by the underlying TTS engine. For
        KaniTTS this maps to the lowercase speaker token that is prepended to
        the text prompt.
    display_name:
        Human readable name used in the UI and API responses.
    language:
        ISO-like language tag that indicates the primary language of the
        speaker. Purely informational for the current implementation.
    description:
        Free-form description to help users pick an appropriate voice.
    """

    identifier: str
    display_name: str
    language: str
    description: str


VOICE_PRESETS: List[VoiceProfile] = [
    VoiceProfile("andrew", "Andrew (English)", "en", "Neutral North American narrator."),
    VoiceProfile("jenny", "Jenny (English, Irish)", "en", "Expressive female voice with an Irish accent."),
    VoiceProfile("david", "David (English, British)", "en", "Warm British male narrator."),
    VoiceProfile("karim", "Karim (Arabic)", "ar", "Modern Standard Arabic male voice."),
    VoiceProfile("maria", "Maria (Spanish)", "es", "Castilian Spanish female narrator."),
    VoiceProfile("seulgi", "Seulgi (Korean)", "ko", "Clear Korean female voice."),
    VoiceProfile("thorsten", "Thorsten (German)", "de", "German male voice suitable for narration."),
]


DEFAULT_ADVANCED_PARAMS: Dict[str, float] = {
    "temperature": 1.4,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "max_new_tokens": 1200,
}


def get_voice_choices() -> List[str]:
    """Return display names for UI dropdowns."""

    return [voice.display_name for voice in VOICE_PRESETS]


def get_voice_by_display_name(display_name: str) -> VoiceProfile:
    """Resolve a display name to its voice profile.

    Parameters
    ----------
    display_name:
        The UI facing voice label.

    Raises
    ------
    KeyError
        If no matching voice is registered.
    """

    for voice in VOICE_PRESETS:
        if voice.display_name == display_name:
            return voice
    raise KeyError(f"Unknown voice: {display_name!r}")


def get_voice_by_identifier(identifier: str) -> VoiceProfile:
    """Resolve a raw identifier to its profile."""

    for voice in VOICE_PRESETS:
        if voice.identifier == identifier:
            return voice
    raise KeyError(f"Unknown voice identifier: {identifier!r}")
