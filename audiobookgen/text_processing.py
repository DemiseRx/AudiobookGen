"""Utilities for handling text inputs and file extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import docx  # type: ignore


class TextExtractionError(RuntimeError):
    """Raised when we cannot extract text from a provided file."""


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx"}


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Read a text file, returning its contents."""

    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16")


def read_docx_file(path: Path) -> str:
    """Extract text from a Word document using python-docx."""

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # pragma: no cover - library errors
        raise TextExtractionError(f"Failed to read {path}: {exc}") from exc

    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def extract_text_from_file(path: Path) -> str:
    """Extract textual content from a supported file type."""

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise TextExtractionError(f"Unsupported file type: {extension}")

    if extension in {".txt", ".md"}:
        return read_text_file(path)

    if extension == ".docx":
        return read_docx_file(path)

    raise TextExtractionError(f"No extractor implemented for {extension}")


def iter_nonempty_segments(text: str) -> Iterable[str]:
    """Yield non-empty trimmed segments from the raw text."""

    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            yield candidate


def normalize_newlines(text: str) -> str:
    """Ensure newline formatting is consistent for downstream splitting."""

    return "\n".join(iter_nonempty_segments(text))


def summarize_text(text: str, max_chars: int = 80) -> str:
    """Return a short summary of the provided text for logging."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."
