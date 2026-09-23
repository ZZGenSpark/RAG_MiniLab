"""Helpers for tests that talk to a live Ollama server."""

from __future__ import annotations

import httpx


def ollama_connection_error(exc: BaseException) -> bool:
    """Return whether the error means Ollama could not be reached."""
    return isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError))
