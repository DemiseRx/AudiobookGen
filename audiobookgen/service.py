"""High level orchestration for the AudiobookGen service."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .chunker import TextChunk, chunk_text, chunk_text_for_manual_mode
from .text_processing import normalize_newlines, summarize_text
from .tts_engine import GenerationConfig, KaniTTSEngine
from .voices import DEFAULT_VOICE, available_voice_ids

LOGGER = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Represents the outcome of generating one or more audio chunks."""

    audio_paths: List[Path]
    combined_path: Path
    total_duration: float
    segments: List[TextChunk]


@dataclass
class ManualSession:
    """Stateful representation of a manual mode synthesis workflow."""

    session_id: str
    chunks: List[TextChunk]
    generated_paths: Dict[int, Path] = field(default_factory=dict)

    def remaining_chunks(self) -> Iterable[TextChunk]:
        for chunk in self.chunks:
            if chunk.index not in self.generated_paths:
                yield chunk


class SynthesisService:
    """Coordinate text preparation, chunking, and waveform generation."""

    def __init__(
        self,
        output_dir: Path = Path("outputs"),
        engine: Optional[KaniTTSEngine] = None,
    ) -> None:
        self.output_dir = output_dir
        self.engine = engine or KaniTTSEngine()
        self._manual_sessions: Dict[str, ManualSession] = {}

    def generate_audio(
        self,
        text: str,
        speaker: str = DEFAULT_VOICE,
        config: Optional[GenerationConfig] = None,
        manual: bool = False,
    ) -> SynthesisResult:
        """Entry point for automatic or manual synthesis."""

        sanitized = normalize_newlines(text)
        tokenizer = self.engine.tokenizer
        if manual:
            chunks = chunk_text_for_manual_mode(sanitized, tokenizer)
        else:
            chunks = chunk_text(sanitized, tokenizer)

        LOGGER.info("Prepared %s chunks for synthesis", len(chunks))
        start_time = time.time()
        paths: List[Path] = []
        for chunk in chunks:
            waveform = self.engine.synthesize(chunk.text, speaker=speaker, config=config)
            filename = f"{int(start_time)}_{chunk.index:03d}.wav"
            path = self.engine.save_waveform(waveform, self.output_dir / filename)
            paths.append(path)
            LOGGER.info("Generated chunk %s (%s tokens) -> %s", chunk.index, chunk.token_length, path)

        combined_path = self._combine_paths(paths)
        total_duration = time.time() - start_time
        return SynthesisResult(paths, combined_path, total_duration, chunks)

    def start_manual_session(self, text: str, speaker: str = DEFAULT_VOICE) -> ManualSession:
        tokenizer = self.engine.tokenizer
        chunks = chunk_text_for_manual_mode(normalize_newlines(text), tokenizer)
        session_id = uuid.uuid4().hex
        session = ManualSession(session_id=session_id, chunks=chunks)
        self._manual_sessions[session_id] = session
        LOGGER.info("Started manual session %s with %s chunks", session_id, len(chunks))
        return session

    def synthesize_manual_chunk(
        self,
        session_id: str,
        index: int,
        speaker: str = DEFAULT_VOICE,
        config: Optional[GenerationConfig] = None,
    ) -> Path:
        session = self._manual_sessions[session_id]
        chunk = next((c for c in session.chunks if c.index == index), None)
        if chunk is None:
            raise ValueError(f"Chunk {index} not found in session {session_id}")
        waveform = self.engine.synthesize(chunk.text, speaker=speaker, config=config)
        filename = f"{session_id}_{index:03d}.wav"
        path = self.engine.save_waveform(waveform, self.output_dir / session_id / filename)
        session.generated_paths[index] = path
        LOGGER.info("Manual session %s generated chunk %s -> %s", session_id, index, path)
        return path

    def _combine_paths(self, paths: Iterable[Path]) -> Path:
        import soundfile as sf

        combined_waveform: Optional[np.ndarray] = None
        for path in paths:
            data, _ = sf.read(str(path))
            combined_waveform = data if combined_waveform is None else np.concatenate([combined_waveform, data])
        combined_path = self.output_dir / "combined.wav"
        if combined_waveform is not None:
            sf.write(str(combined_path), combined_waveform, self.engine.waveform_collector.sample_rate)
        return combined_path

    def list_available_voices(self) -> Iterable[str]:
        return list(available_voice_ids())
