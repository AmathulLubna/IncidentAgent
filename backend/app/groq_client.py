"""
Thin wrapper around Groq's OpenAI-compatible chat completions endpoint.

Tries primary model first; on API error or malformed tool-call arguments
retries once against the fallback model. Failures are captured on the
returned envelope with full observability (model + exception type + message)
logged server-side, while the user-facing message stays safe.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import APIError, OpenAI

from .config import settings

logger = logging.getLogger("groq_client")


@dataclass
class ChatResult:
    message: Optional[Any] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    used_fallback: bool = False
    primary_error: Optional[str] = None
    fallback_error: Optional[str] = None


def _client() -> OpenAI:
    return OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROQ_BASE_URL)


def _validate_tool_calls(message) -> bool:
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        try:
            json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, AttributeError):
            return False
    return True


def _attempt(model: str, messages: list[dict], tools: Optional[list[dict]], tool_choice) -> Any:
    client = _client()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def chat_completion(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = "auto",
) -> ChatResult:
    primary_error: Optional[str] = None
    # --- Attempt 1: primary ---
    try:
        message = _attempt(settings.GROQ_MODEL_PRIMARY, messages, tools, tool_choice)
        if not _validate_tool_calls(message):
            raise ValueError("Primary model returned malformed tool-call arguments.")
        return ChatResult(message=message, model_used=settings.GROQ_MODEL_PRIMARY)
    except Exception as primary_err:  # noqa: BLE001 - deliberately broad, see fallback
        primary_error = f"{type(primary_err).__name__}: {primary_err}"
        logger.warning(
            "Groq primary failed: model=%s error_type=%s error=%s",
            settings.GROQ_MODEL_PRIMARY, type(primary_err).__name__, primary_err,
        )

    # --- Attempt 2: fallback ---
    try:
        message = _attempt(settings.GROQ_MODEL_FALLBACK, messages, tools, tool_choice)
        if not _validate_tool_calls(message):
            fb = "Fallback model also returned malformed tool-call arguments."
            logger.error(
                "Groq fallback failed: model=%s error_type=ValueError error=%s",
                settings.GROQ_MODEL_FALLBACK, fb,
            )
            return ChatResult(error=fb, used_fallback=True, primary_error=primary_error,
                              fallback_error=f"ValueError: {fb}")
        return ChatResult(message=message, model_used=settings.GROQ_MODEL_FALLBACK,
                          used_fallback=True, primary_error=primary_error)
    except Exception as fallback_err:  # noqa: BLE001
        fb_msg = f"{type(fallback_err).__name__}: {fallback_err}"
        logger.error(
            "Groq fallback failed: model=%s error_type=%s error=%s",
            settings.GROQ_MODEL_FALLBACK, type(fallback_err).__name__, fallback_err,
        )
        return ChatResult(error=str(fallback_err), used_fallback=True,
                          primary_error=primary_error, fallback_error=fb_msg)


async def achat_completion(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = "auto",
) -> ChatResult:
    """Async wrapper — runs the blocking OpenAI client in a thread."""
    return await asyncio.to_thread(chat_completion, messages, tools, tool_choice)


def groq_health() -> dict:
    """Lightweight connectivity check (no tools). Returns status dict, never raises."""
    if not settings.GROQ_API_KEY:
        return {"status": "missing_key"}
    try:
        msg = _attempt(
            settings.GROQ_MODEL_PRIMARY,
            [{"role": "user", "content": "Reply with exactly: ok"}],
            None, None,
        )
        content = (getattr(msg, "content", "") or "").strip().lower()
        return {"status": "ok", "model": settings.GROQ_MODEL_PRIMARY, "reply": content[:50]}
    except Exception as e:  # noqa: BLE001
        logger.warning("groq_health primary failed: %s: %s", type(e).__name__, e)
        try:
            msg = _attempt(
                settings.GROQ_MODEL_FALLBACK,
                [{"role": "user", "content": "Reply with exactly: ok"}],
                None, None,
            )
            content = (getattr(msg, "content", "") or "").strip().lower()
            return {"status": "fallback_ok", "model": settings.GROQ_MODEL_FALLBACK,
                    "reply": content[:50], "primary_error": f"{type(e).__name__}: {e}"}
        except Exception as e2:  # noqa: BLE001
            return {"status": "unreachable",
                    "primary_error": f"{type(e).__name__}: {e}",
                    "fallback_error": f"{type(e2).__name__}: {e2}"}
