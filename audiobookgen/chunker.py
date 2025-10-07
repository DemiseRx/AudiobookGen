"""Utility helpers for splitting long texts into manageable synthesis chunks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TextChunk:
    """Represents a single synthesis unit."""

    index: int
    text: str

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text}


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def normalise_whitespace(text: str) -> str:
    """Collapse superfluous whitespace while keeping intentional paragraph gaps."""

    cleaned = re.sub(r"\s+", " ", text.strip())
    # Reintroduce double new lines for paragraph style separation where possible.
    cleaned = cleaned.replace(" \n ", "\n\n")
    return cleaned


def split_text(
    text: str,
    *,
    max_words: int = 220,
    max_characters: int = 1500,
) -> List[TextChunk]:
    """Split ``text`` into natural sounding chunks.

    The splitter favours paragraph and sentence boundaries while keeping the
    resulting segments within soft word/character limits. These defaults work
    well for KaniTTS which prefers prompts under roughly two thousand tokens.
    """

    if not text.strip():
        return []

    paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
    chunks: List[TextChunk] = []
    current_chunk: List[str] = []
    word_count = 0

    def flush() -> None:
        nonlocal word_count, current_chunk, chunks
        if not current_chunk:
            return
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(TextChunk(len(chunks), chunk_text))
        current_chunk = []
        word_count = 0

    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            if not sentence:
                continue
            sentence_words = sentence.split()
            prospective_words = word_count + len(sentence_words)
            prospective_chars = sum(len(part) for part in current_chunk) + len(sentence)
            if (
                prospective_words > max_words
                or prospective_chars > max_characters
            ) and current_chunk:
                flush()
            current_chunk.append(sentence)
            word_count += len(sentence_words)
        flush()

    flush()
    if not chunks:
        chunks.append(TextChunk(0, text.strip()))
    return chunks


def estimate_total_audio_duration(word_count: int, words_per_minute: int = 150) -> float:
    """Approximate duration (in seconds) for a given ``word_count``."""

    minutes = word_count / max(words_per_minute, 1)
    return minutes * 60


def chunks_from_iterable(lines: Iterable[str], **kwargs) -> List[TextChunk]:
    """Load text from an iterator before chunking."""

    return split_text("\n".join(lines), **kwargs)
