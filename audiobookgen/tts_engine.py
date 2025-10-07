"""Implementation of the KaniTTS synthesis backend."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nemo.collections.asr.parts.submodules import WaveformCollector  # type: ignore
from nemo.collections.tts.models import AudioCodecModel  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

from .voices import DEFAULT_VOICE

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for sampling from KaniTTS."""

    temperature: float = 1.4
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1200


class KaniTTSEngine:
    """Wrapper around the Hugging Face KaniTTS model and NVIDIA codec."""

    def __init__(
        self,
        model_id: str = "nineninesix/kani-tts-370m",
        codec_id: str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.codec_id = codec_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = None
        self._model = None
        self._codec = None
        self._waveform_collector: Optional[WaveformCollector] = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            LOGGER.info("Loading tokenizer %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            LOGGER.info("Loading language model %s", self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            self._model.to(self.device)
        return self._model

    @property
    def codec(self):
        if self._codec is None:
            LOGGER.info("Loading codec %s", self.codec_id)
            self._codec = AudioCodecModel.from_pretrained(self.codec_id).to(self.device)
        return self._codec

    @property
    def waveform_collector(self) -> WaveformCollector:
        if self._waveform_collector is None:
            self._waveform_collector = WaveformCollector(sample_rate=22050)
        return self._waveform_collector

    def _prepare_prompt(self, text: str, speaker: Optional[str]) -> str:
        speaker_id = speaker or DEFAULT_VOICE
        return f"{speaker_id}: {text}" if not text.lower().startswith(f"{speaker_id}:") else text

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> np.ndarray:
        """Generate an audio waveform for the provided text."""

        generation_config = config or GenerationConfig()
        tokenizer = self.tokenizer
        prompt = self._prepare_prompt(text, speaker)
        encoded = tokenizer(prompt, return_tensors="pt").to(self.device)

        LOGGER.debug("Generating audio tokens for %s characters", len(text))
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                repetition_penalty=generation_config.repetition_penalty,
                max_new_tokens=generation_config.max_new_tokens,
            )

        tokens = output[0].detach().cpu().numpy()
        audio_tokens = self._extract_audio_tokens(tokens)
        return self._decode_audio_tokens(audio_tokens)

    def _extract_audio_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Slice the generated ids to obtain audio codec tokens."""

        tokenizer = self.tokenizer
        start_token = tokenizer.convert_tokens_to_ids("<|startofvoice|>")
        end_token = tokenizer.eos_token_id

        try:
            start_index = np.where(tokens == start_token)[0][0] + 1
        except IndexError as exc:
            raise RuntimeError("Generated sequence missing start-of-voice marker") from exc

        end_candidates = np.where(tokens == end_token)[0]
        if len(end_candidates) == 0:
            raise RuntimeError("Generated sequence missing end-of-speech marker")
        end_index = end_candidates[0]
        return tokens[start_index:end_index]

    def _decode_audio_tokens(self, audio_tokens: np.ndarray) -> np.ndarray:
        """Decode audio tokens to a waveform array."""

        codec = self.codec
        length_tensor = torch.tensor([audio_tokens.shape[0]], device=self.device)
        tokens_tensor = torch.tensor(audio_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            waveform = codec.decode(tokens=tokens_tensor, tokens_len=length_tensor)
        return waveform.squeeze().cpu().numpy()

    def save_waveform(self, waveform: np.ndarray, path: Path) -> Path:
        """Persist waveform data to a WAV file."""

        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), waveform, self.waveform_collector.sample_rate)
        return path
