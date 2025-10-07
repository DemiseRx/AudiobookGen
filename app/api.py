"""FastAPI application exposing both HTML UI and JSON endpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audiobookgen.config import DEFAULT_ADVANCED_PARAMS, VOICE_PRESETS, get_voice_by_display_name
from audiobookgen.service import AudiobookService, SynthesisRequest, create_service

app = FastAPI(title="AudiobookGen", version="0.1.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependency injection ---------------------------------------------------------

SERVICE = create_service()


def get_service() -> AudiobookService:
    return SERVICE


# Utility helpers --------------------------------------------------------------

def _load_text_from_file(upload: UploadFile) -> str:
    raw = upload.file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


async def _payload_from_form(
    text: str = Form(""),
    voice: str = Form(VOICE_PRESETS[0].display_name),
    temperature: float = Form(DEFAULT_ADVANCED_PARAMS["temperature"]),
    top_p: float = Form(DEFAULT_ADVANCED_PARAMS["top_p"]),
    repetition_penalty: float = Form(DEFAULT_ADVANCED_PARAMS["repetition_penalty"]),
    max_new_tokens: int = Form(DEFAULT_ADVANCED_PARAMS["max_new_tokens"]),
    mode: str = Form("automatic"),
    upload: Optional[UploadFile] = File(None),
) -> Dict[str, str]:
    file_text = ""
    if upload is not None:
        file_text = _load_text_from_file(upload)
    payload = {
        "text": text,
        "file_text": file_text,
        "voice": voice,
        "temperature": str(temperature),
        "top_p": str(top_p),
        "repetition_penalty": str(repetition_penalty),
        "max_new_tokens": str(max_new_tokens),
        "mode": mode,
    }
    return payload


# HTML routes ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "voices": VOICE_PRESETS,
            "defaults": DEFAULT_ADVANCED_PARAMS,
        },
    )


@app.post("/synthesize", response_class=HTMLResponse)
async def synthesize_from_form(
    request: Request,
    payload: Dict[str, str] = Depends(_payload_from_form),
    service: AudiobookService = Depends(get_service),
) -> HTMLResponse:
    text = (payload.get("text") or payload.get("file_text") or "").strip()
    if not text:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "voices": VOICE_PRESETS,
                "defaults": DEFAULT_ADVANCED_PARAMS,
                "error": "Please provide text or upload a file.",
            },
            status_code=400,
        )
    voice = get_voice_by_display_name(payload["voice"])
    request_obj = SynthesisRequest(
        text=text,
        voice=voice,
        advanced_params={
            "temperature": float(payload["temperature"]),
            "top_p": float(payload["top_p"]),
            "repetition_penalty": float(payload["repetition_penalty"]),
            "max_new_tokens": int(float(payload["max_new_tokens"])),
        },
    )
    if payload["mode"] == "manual":
        session = service.create_manual_session(text, voice, request_obj.advanced_params)
        return templates.TemplateResponse(
            "manual_session.html",
            {
                "request": request,
                "session": session,
                "voices": VOICE_PRESETS,
            },
        )

    result = service.synthesise_automatic(request_obj)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
        },
    )


@app.post("/manual/{session_id}/segment/{segment_index}")
async def manual_segment(
    session_id: str,
    segment_index: int,
    voice: str = Form(""),
    service: AudiobookService = Depends(get_service),
) -> RedirectResponse:
    session = service.sessions.get(session_id)
    voice_profile = get_voice_by_display_name(voice) if voice else session.voice
    service.synthesise_manual_segment(session_id, segment_index, voice_profile)
    return RedirectResponse(url=f"/manual/{session_id}", status_code=303)


@app.get("/manual/{session_id}", response_class=HTMLResponse)
async def manual_session_view(
    request: Request,
    session_id: str,
    service: AudiobookService = Depends(get_service),
) -> HTMLResponse:
    session = service.sessions.get(session_id)
    return templates.TemplateResponse(
        "manual_session.html",
        {
            "request": request,
            "session": session,
            "voices": VOICE_PRESETS,
        },
    )


# JSON API ---------------------------------------------------------------------


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@app.post("/api/synthesize")
async def api_synthesize(payload: Dict[str, str]) -> JSONResponse:
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    voice_name = payload.get("voice", VOICE_PRESETS[0].display_name)
    voice = get_voice_by_display_name(voice_name)
    advanced = {**DEFAULT_ADVANCED_PARAMS}
    for key in advanced.keys():
        if key in payload:
            advanced[key] = float(payload[key]) if key != "max_new_tokens" else int(float(payload[key]))
    service = get_service()
    request_obj = SynthesisRequest(text=text, voice=voice, advanced_params=advanced)
    result = service.synthesise_automatic(request_obj)
    return JSONResponse(
        {
            "audio_file": str(result.output_path),
            "duration_seconds": result.duration_seconds,
            "segments": [segment.index for segment in result.segments],
            "voice": voice.display_name,
        }
    )


@app.get("/download/{file_name}")
async def download(file_name: str) -> FileResponse:
    path = Path("outputs") / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}
