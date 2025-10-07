"""High level orchestration for the audiobook generation workflow."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from .chunker import TextChunk, split_text
from .config import DEFAULT_ADVANCED_PARAMS, VoiceProfile, get_voice_by_display_name
from .tts import BaseTTSEngine, SynthesisSegment, load_engine


@dataclass
class SynthesisRequest:
    text: str
    voice: VoiceProfile
    advanced_params: Dict[str, float]
    session_id: Optional[str] = None


@dataclass
class SynthesisResult:
    output_path: Path
    duration_seconds: float
    segments: List[SynthesisSegment] = field(default_factory=list)


@dataclass
class ManualSession:
    session_id: str
    segments: List[TextChunk]
    voice: VoiceProfile
    advanced_params: Dict[str, float]
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())
    generated_segments: Dict[int, Path] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id,
            "voice": self.voice.display_name,
            "segments": json.dumps([chunk.to_dict() for chunk in self.segments]),
        }


class ManualSessionManager:
    """In-memory session store for manual workflows."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ManualSession] = {}

    def create(
        self,
        chunks: Iterable[TextChunk],
        voice: VoiceProfile,
        advanced_params: Dict[str, float],
    ) -> ManualSession:
        session_id = uuid.uuid4().hex
        session = ManualSession(
            session_id=session_id,
            segments=list(chunks),
            voice=voice,
            advanced_params=advanced_params,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> ManualSession:
        return self._sessions[session_id]

    def list_active(self) -> List[ManualSession]:
        return list(self._sessions.values())


class AudiobookService:
    """Entry point used by both the API and the HTML interface."""

    def __init__(
        self,
        engine: Optional[BaseTTSEngine] = None,
        output_dir: Path | str = "outputs",
    ) -> None:
        self.engine = engine or load_engine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = ManualSessionManager()

    # ------------------------------------------------------------------
    # Automatic workflow
    # ------------------------------------------------------------------
    def synthesise_automatic(self, request: SynthesisRequest) -> SynthesisResult:
        chunks = split_text(request.text)
        segments: List[SynthesisSegment] = []
        for chunk in chunks:
            audio = self.engine.synthesise(
                chunk.text,
                request.voice.identifier,
                **request.advanced_params,
            )
            segments.append(
                SynthesisSegment(
                    index=chunk.index,
                    text=chunk.text,
                    audio=audio,
                    sample_rate=self.engine.sample_rate,
                    voice_id=request.voice.identifier,
                )
            )

        output_path = self._write_segments_to_file(segments, request.session_id)
        duration_seconds = sum(len(seg.audio) for seg in segments) / self.engine.sample_rate
        return SynthesisResult(output_path=output_path, duration_seconds=duration_seconds, segments=segments)

    # ------------------------------------------------------------------
    # Manual workflow
    # ------------------------------------------------------------------
    def create_manual_session(
        self,
        text: str,
        voice: VoiceProfile,
        advanced_params: Dict[str, float],
    ) -> ManualSession:
        chunks = split_text(text)
        return self.sessions.create(chunks, voice, advanced_params)

    def synthesise_manual_segment(
        self, session_id: str, segment_index: int, voice_override: Optional[VoiceProfile] = None
    ) -> Path:
        session = self.sessions.get(session_id)
        chunk = session.segments[segment_index]
        voice = voice_override or session.voice
        audio = self.engine.synthesise(
            chunk.text,
            voice.identifier,
            **session.advanced_params,
        )
        output_path = self._write_audio_chunk(audio, session_id, segment_index, voice.identifier)
        session.generated_segments[segment_index] = output_path
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write_segments_to_file(
        self,
        segments: Iterable[SynthesisSegment],
        session_id: Optional[str] = None,
    ) -> Path:
        audio = np.concatenate([segment.audio for segment in segments])
        file_name = self._build_output_name(session_id)
        output_path = self.output_dir / file_name
        sf.write(output_path, audio, self.engine.sample_rate)
        return output_path

    def _write_audio_chunk(
        self,
        audio: np.ndarray,
        session_id: str,
        segment_index: int,
        voice_id: str,
    ) -> Path:
        file_name = f"{session_id}_segment_{segment_index:04d}_{voice_id}.wav"
        output_path = self.output_dir / file_name
        sf.write(output_path, audio, self.engine.sample_rate)
        return output_path

    def _build_output_name(self, session_id: Optional[str] = None) -> str:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if session_id:
            return f"audiobook_{session_id}_{timestamp}.wav"
        return f"audiobook_{timestamp}.wav"


def create_service(engine: Optional[BaseTTSEngine] = None) -> AudiobookService:
    return AudiobookService(engine=engine)


def default_request_from_payload(payload: Dict[str, str]) -> SynthesisRequest:
    text = payload.get("text", "")
    if not text and payload.get("file_text"):
        text = payload["file_text"]
    voice_display = payload.get("voice")
    voice = get_voice_by_display_name(voice_display) if voice_display else VoiceProfile("", "Default", "", "")
    advanced = {**DEFAULT_ADVANCED_PARAMS}
    for key in ("temperature", "top_p", "repetition_penalty", "max_new_tokens"):
        if key in payload and payload[key]:
            advanced[key] = float(payload[key])
    return SynthesisRequest(text=text, voice=voice, advanced_params=advanced)
