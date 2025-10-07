"""Voice configuration for KaniTTS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class VoiceProfile:
    """Metadata about a KaniTTS voice option."""

    speaker_id: str
    language: str
    description: str


_VOICES: Dict[str, VoiceProfile] = {
    "andrew": VoiceProfile("andrew", "English", "Neutral US male narrator"),
    "david": VoiceProfile("david", "English", "British male narrator"),
    "jenny": VoiceProfile("jenny", "English", "Irish female narrator"),
    "kara": VoiceProfile("kara", "English", "North American female narrator"),
    "liang": VoiceProfile("liang", "Chinese", "Mandarin male narrator"),
    "maria": VoiceProfile("maria", "Spanish", "Spanish female narrator"),
    "seulgi": VoiceProfile("seulgi", "Korean", "Korean female narrator"),
    "thorsten": VoiceProfile("thorsten", "German", "German male narrator"),
    "karim": VoiceProfile("karim", "Arabic", "Arabic male narrator"),
}


DEFAULT_VOICE = "andrew"


def available_voice_ids() -> Iterable[str]:
    """Return the iterable of available speaker identifiers."""

    return _VOICES.keys()


def get_voice_profile(voice_id: str) -> VoiceProfile:
    """Return the profile for a voice ID, raising if not present."""

    try:
        return _VOICES[voice_id]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown voice id: {voice_id}") from exc


def voice_choices() -> List[str]:
    """Return human readable options for UI dropdowns."""

    return [f"{profile.speaker_id} ({profile.language}) - {profile.description}" for profile in _VOICES.values()]
