"""TTS engine abstraction used by the service.

The project is designed to support the KaniTTS model, but the default
implementation ships with a lightweight mock engine so that the remainder of the
system can be exercised without requiring a GPU or downloading the large
checkpoint. Deployments with access to the real model can plug in an engine that
wraps the official Hugging Face code path.
"""

from __future__ import annotations

import contextlib
import importlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EngineConfig:
    """Configuration object passed to :class:`BaseTTSEngine`."""

    sample_rate: int = 22050
    amplitude: float = 0.2


@dataclass
class SynthesisSegment:
    """Represents the outcome for a single text chunk."""

    index: int
    text: str
    audio: np.ndarray
    sample_rate: int
    voice_id: str


class BaseTTSEngine:
    """Interface implemented by concrete engines."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def synthesise(self, text: str, voice_id: str, **kwargs) -> np.ndarray:
        """Return a waveform for ``text`` using ``voice_id``.

        Sub-classes must override this method. The base implementation raises an
        informative error so that integrators know they need to supply a
        concrete backend.
        """

        raise NotImplementedError("No TTS engine has been configured")


class MockTTSEngine(BaseTTSEngine):
    """Simple engine that generates a synthetic tone for demonstration.

    The waveform encodes a deterministic mapping between the text length, voice
    identifier and advanced parameters. While it does not perform real speech
    synthesis it is extremely useful for unit tests, API prototyping and for
    illustrating how the remainder of the stack behaves.
    """

    def synthesise(self, text: str, voice_id: str, **kwargs) -> np.ndarray:
        if not text:
            return np.zeros(1, dtype=np.float32)

        duration_seconds = max(len(text.split()) / 2.5, 0.5)
        num_samples = int(duration_seconds * self.sample_rate)
        time = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        base_freq = 220 + (hash(voice_id) % 200)
        modulation = sum(float(v) for v in kwargs.values() if isinstance(v, (int, float)))
        waveform = np.sin(2 * math.pi * (base_freq + modulation) * time)
        waveform *= self.config.amplitude
        return waveform.astype(np.float32)


def load_engine(prefer_real_engine: bool = True) -> BaseTTSEngine:
    """Attempt to load the real KaniTTS engine, fallback to :class:`MockTTSEngine`.

    The function keeps the heavy dependencies optional which keeps the example
    project light-weight. If the required packages are available the loader will
    instantiate a :class:`KaniEngine` wrapper that mirrors the Hugging Face
    reference implementation.
    """

    if prefer_real_engine:
        with contextlib.suppress(Exception):
            return _load_kani_engine()
    return MockTTSEngine()


def _load_kani_engine() -> BaseTTSEngine:
    """Dynamically import the official KaniTTS stack.

    Users that want actual speech output should install the ``transformers`` and
    ``nemo_toolkit`` packages alongside the GPU dependencies. When these
    dependencies are available the helper initialises the models lazily and
    exposes them through a very small adapter class.
    """

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    nemo_models = importlib.import_module("nemo.collections.tts.models.audio_codec")

    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AudioCodecModel = nemo_models.AudioCodecModel

    tokenizer = AutoTokenizer.from_pretrained(
        "nineninesix/kani-tts-370m", trust_remote_code=True
    )
    tts_model = AutoModelForCausalLM.from_pretrained(
        "nineninesix/kani-tts-370m",
        trust_remote_code=True,
        torch_dtype=getattr(torch, "bfloat16", torch.float16),
        device_map="auto",
    )
    codec_model = AudioCodecModel.from_pretrained(
        "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    )

    class KaniEngine(BaseTTSEngine):
        def __init__(self) -> None:
            super().__init__(EngineConfig(sample_rate=22050))
            self._tokenizer = tokenizer
            self._tts_model = tts_model
            self._codec = codec_model.to(self._tts_model.device)
            self._start_of_speech = len(self._tokenizer) + 1
            self._end_of_speech = len(self._tokenizer) + 2

        def synthesise(self, text: str, voice_id: str, **kwargs) -> np.ndarray:
            if not text:
                return np.zeros(1, dtype=np.float32)
            prompt = f"{voice_id}: {text}" if voice_id else text
            inputs = self._tokenizer(prompt, return_tensors="pt").to(
                self._tts_model.device
            )
            generation_config = {
                "do_sample": True,
                "temperature": kwargs.get("temperature", 1.4),
                "top_p": kwargs.get("top_p", 0.95),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "max_new_tokens": kwargs.get("max_new_tokens", 1200),
                "eos_token_id": self._end_of_speech,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            with torch.inference_mode():
                output_tokens = self._tts_model.generate(
                    **inputs, **generation_config
                )[0].detach().cpu().numpy()
            try:
                start_idx = list(output_tokens).index(self._start_of_speech)
                end_idx = list(output_tokens).index(self._end_of_speech, start_idx + 1)
            except ValueError as exc:  # pragma: no cover - indicates truncated output
                raise RuntimeError("KaniTTS did not emit audio tokens") from exc
            audio_tokens = output_tokens[start_idx + 1 : end_idx]
            tokens = torch.tensor(audio_tokens, dtype=torch.long).unsqueeze(0).to(
                self._tts_model.device
            )
            token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long).to(
                self._tts_model.device
            )
            with torch.inference_mode():
                waveform = self._codec.decode(tokens=tokens, tokens_len=token_lengths)
            return waveform.cpu().numpy().flatten().astype(np.float32)

    return KaniEngine()
