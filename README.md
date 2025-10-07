# AudiobookGen

AudiobookGen is a local-first text-to-speech service that wraps the KaniTTS
model in a web UI and REST API. The project supports two operational modes:

- **Automatic mode** – one click to process the entire input using the default
  voice and parameters.
- **Manual mode** – advanced operators can inspect detected text chunks and
  synthesise them one-by-one, optionally using different voices per segment.

The repository is structured so that the heavy KaniTTS dependency tree is
optional. By default a lightweight mock engine generates placeholder audio which
keeps the project runnable in constrained environments while preserving the
control flow. Deployments that require real speech can install the official
KaniTTS model (see below) and the service will use it automatically.

## Features

- Multi-voice selection based on the KaniTTS presets
- Text box and file upload input methods
- Automatic chunking for long texts
- REST endpoint for programmatic integrations (`POST /api/synthesize`)
- Manual segmentation workflow with per-chunk downloads

## Getting started

### Requirements

- Python 3.10+
- (Optional for real speech) CUDA-capable GPU with ~2&nbsp;GB VRAM and the
  `transformers`, `torch`, and `nemo_toolkit` packages installed

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the server

```bash
python -m app.server
```

The service exposes an interactive UI at `http://localhost:8000` and the JSON
API at `http://localhost:8000/api/synthesize`.

### Using the API

Example request using `curl`:

```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Once upon a midnight dreary...",
        "voice": "Jenny (English, Irish)",
        "temperature": 1.2
      }'
```

Response:

```json
{
  "audio_file": "outputs/audiobook_20240101_120000.wav",
  "duration_seconds": 18.4,
  "segments": [0, 1, 2],
  "voice": "Jenny (English, Irish)"
}
```

The path refers to a WAV file stored inside the `outputs/` directory.

## Enabling the real KaniTTS model

Install the additional dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers nemo_toolkit[tts] librosa
```

When these packages are available the service automatically loads the official
KaniTTS weights during startup. Otherwise, it falls back to the deterministic
mock engine to keep the workflow testable.

## Development

- Static files live in `static/`, templates in `templates/`, and the business
  logic in the `audiobookgen/` package.
- Outputs are stored in the `outputs/` directory by default.
- Run `python -m compileall .` to ensure there are no syntax errors.

## License

MIT
