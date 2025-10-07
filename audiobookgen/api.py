"""FastAPI application exposing the AudiobookGen service."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .service import SynthesisService
from .text_processing import TextExtractionError, extract_text_from_file
from .tts_engine import GenerationConfig
from .voices import DEFAULT_VOICE

LOGGER = logging.getLogger(__name__)


class SynthesisRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw text to convert to speech")
    voice: str = Field(DEFAULT_VOICE, description="Speaker identifier")
    temperature: float = Field(1.4, ge=0.1, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=0.8, le=2.0)
    manual: bool = Field(False, description="If True, use manual segmentation settings")


class SynthesisResponse(BaseModel):
    audio_file: str
    chunks: int
    duration_seconds: float
    chunk_files: List[str]


class ManualSessionCreate(BaseModel):
    text: str
    voice: str = Field(DEFAULT_VOICE)


class ManualChunkResponse(BaseModel):
    session_id: str
    chunk_index: int
    audio_file: str


class ManualSessionStatus(BaseModel):
    session_id: str
    pending_chunks: List[int]
    completed_chunks: List[int]


def get_service() -> SynthesisService:
    return SynthesisService()


def create_app(service: SynthesisService | None = None) -> FastAPI:
    app = FastAPI(title="AudiobookGen", description="KaniTTS synthesis service")
    service_instance = service or SynthesisService()

    @app.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize_text(payload: SynthesisRequest, svc: SynthesisService = Depends(lambda: service_instance)):
        text = payload.text
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        config = GenerationConfig(
            temperature=payload.temperature,
            top_p=payload.top_p,
            repetition_penalty=payload.repetition_penalty,
        )
        LOGGER.info("API synthesis request voice=%s manual=%s", payload.voice, payload.manual)
        result = svc.generate_audio(text, speaker=payload.voice, config=config, manual=payload.manual)
        return SynthesisResponse(
            audio_file=str(result.combined_path),
            chunks=len(result.segments),
            duration_seconds=result.total_duration,
            chunk_files=[str(path) for path in result.audio_paths],
        )

    @app.post("/synthesize/file", response_model=SynthesisResponse)
    async def synthesize_file(
        voice: str = DEFAULT_VOICE,
        manual: bool = False,
        file: UploadFile = File(...),
        svc: SynthesisService = Depends(lambda: service_instance),
    ):
        try:
            content = await file.read()
            temp_path = Path("/tmp") / f"uploaded_{int(time.time())}_{file.filename}"
            temp_path.write_bytes(content)
            text = extract_text_from_file(temp_path)
        except TextExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()

        result = svc.generate_audio(text, speaker=voice, manual=manual)
        return SynthesisResponse(
            audio_file=str(result.combined_path),
            chunks=len(result.segments),
            duration_seconds=result.total_duration,
            chunk_files=[str(path) for path in result.audio_paths],
        )

    @app.post("/manual/sessions", response_model=ManualSessionStatus)
    async def create_manual_session(payload: ManualSessionCreate, svc: SynthesisService = Depends(lambda: service_instance)):
        session = svc.start_manual_session(payload.text, speaker=payload.voice)
        return ManualSessionStatus(
            session_id=session.session_id,
            pending_chunks=[chunk.index for chunk in session.chunks],
            completed_chunks=[],
        )

    @app.post("/manual/sessions/{session_id}/chunks/{index}", response_model=ManualChunkResponse)
    async def synthesize_manual_chunk(
        session_id: str,
        index: int,
        voice: str = DEFAULT_VOICE,
        svc: SynthesisService = Depends(lambda: service_instance),
    ):
        path = svc.synthesize_manual_chunk(session_id, index, speaker=voice)
        return ManualChunkResponse(session_id=session_id, chunk_index=index, audio_file=str(path))

    @app.get("/manual/sessions/{session_id}", response_model=ManualSessionStatus)
    async def manual_session_status(session_id: str, svc: SynthesisService = Depends(lambda: service_instance)):
        session = svc._manual_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        pending = [chunk.index for chunk in session.remaining_chunks()]
        completed = sorted(session.generated_paths)
        return ManualSessionStatus(session_id=session_id, pending_chunks=pending, completed_chunks=completed)

    return app
