"""Text utility helpers."""

from __future__ import annotations


def compact_text(text: str) -> str:
    """Collapse whitespace into single spaces.

    Args:
        text: Input text.

    Returns:
        Normalized text with collapsed whitespace.
    """
    return " ".join(str(text).split())


def truncate_text(text: str, max_len: int, suffix: str = "...(truncated)") -> str:
    """Truncate text to a maximum length with a suffix.

    Args:
        text: Input text.
        max_len: Maximum length to keep before appending suffix.
        suffix: Suffix appended when truncation occurs.

    Returns:
        Possibly truncated text.
    """
    if len(text) > max_len:
        return f"{text[:max_len]}{suffix}"
    return text
