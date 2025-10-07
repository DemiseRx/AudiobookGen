"""Utilities for splitting long text into model-friendly chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from transformers import PreTrainedTokenizerBase


@dataclass
class TextChunk:
    """Represents a chunk of text ready for synthesis."""

    index: int
    text: str
    token_length: int

    @property
    def safe_prompt(self) -> str:
        """Return text trimmed for logging or manual review."""

        preview = " ".join(self.text.split())
        return preview if len(preview) <= 120 else preview[:117] + "..."


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 1000,
    overlap_sentences: int = 0,
) -> List[TextChunk]:
    """Split the text into manageable pieces based on tokenizer length.

    Args:
        text: Raw text to segment.
        tokenizer: Tokenizer used for counting tokens.
        max_tokens: Maximum tokens per chunk (default 1000 to stay within model limits).
        overlap_sentences: Optional number of trailing sentences to carry over to the
            next chunk. This reduces audible boundary artifacts for long-form speech.
    """

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks: List[TextChunk] = []
    current: List[str] = []

    def flush_chunk(index: int, parts: List[str]) -> None:
        if not parts:
            return
        joined = ". ".join(parts).strip()
        if not joined.endswith("."):
            joined += "."
        token_length = len(tokenizer(joined).input_ids)
        chunks.append(TextChunk(index=index, text=joined, token_length=token_length))

    index = 0
    for sentence in sentences:
        tentative = current + [sentence]
        joined = ". ".join(tentative).strip()
        token_count = len(tokenizer(joined).input_ids)
        if token_count > max_tokens and current:
            flush_chunk(index, current)
            index += 1
            current = tentative[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)

    flush_chunk(index, current)
    return chunks


def chunk_text_for_manual_mode(
    text: str, tokenizer: PreTrainedTokenizerBase, suggested_tokens: int = 600
) -> List[TextChunk]:
    """Generate finer-grained segments for manual oversight."""

    return chunk_text(text, tokenizer, max_tokens=suggested_tokens, overlap_sentences=1)
