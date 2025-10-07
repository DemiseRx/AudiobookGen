# AudiobookGen

AudiobookGen provides a local KaniTTS-powered text-to-speech service with automatic and manual synthesis workflows.

## Features

- **Automatic mode:** one-click synthesis that handles chunking, voice selection, and audio stitching.
- **Manual mode:** advanced control that exposes each chunk for review and allows segment-by-segment voice adjustments.
- **Multi-voice support:** select from predefined KaniTTS speakers.
- **File ingestion:** upload plain text or Word documents; automatic text extraction when the text field is empty.
- **Batch processing:** split long-form content into manageable segments to respect KaniTTS token limits.
- **REST API:** `/synthesize` and `/synthesize/file` endpoints return generated audio files for integration with other tools.

## Running the server

```bash
pip install -r requirements.txt
python main.py
```

The service listens on `http://0.0.0.0:8000` by default.

## API overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/synthesize` | POST | Automatic or manual synthesis from raw text. |
| `/synthesize/file` | POST | Upload a text-based file to synthesize. |
| `/manual/sessions` | POST | Create a manual session and return chunk metadata. |
| `/manual/sessions/{id}` | GET | Inspect remaining and completed chunks. |
| `/manual/sessions/{id}/chunks/{index}` | POST | Generate audio for a specific chunk. |

## Development notes

- Requires access to a CUDA-capable GPU for real-time synthesis.
- Uses Hugging Face Transformers for the KaniTTS model and NVIDIA NeMo for decoding codec tokens.
- Outputs are saved under the `outputs/` directory by default.
- Configure logging level via the `AUDIOBOOKGEN_LOG_LEVEL` environment variable.
