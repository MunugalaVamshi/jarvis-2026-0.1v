
"""
providers/base.py — Local Ollama-only provider contract.

Supported conversation languages:
    English (en), Telugu (te), Hindi (hi).

LLM:
    Ollama local server, normally http://localhost:11434.
    No hosted AI provider integrations.

Speech:
    STT and TTS must use local engines.
    Each provider must enforce the supported-language contract.

The event interface is a typed local event stream consumed directly by
main.py. No hosted/live session protocol is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Ollama and language configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "te", "hi")

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
}

DEFAULT_LANGUAGE = "en"

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"


def normalize_language(language: str | None) -> str:
    """
    Normalize a language code or common language name.

    Unsupported languages fall back to English. STT providers should use
    this function after detecting a language, and reject unsupported audio
    rather than silently pretending it is one of the supported languages.
    """
    if not language:
        return DEFAULT_LANGUAGE

    value = language.strip().lower().replace("_", "-")

    aliases = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "en-us": "en",
        "en-gb": "en",
        "te": "te",
        "tel": "te",
        "telugu": "te",
        "hi": "hi",
        "hin": "hi",
        "hindi": "hi",
    }

    # Accept regional codes such as te-IN and hi-IN.
    if value in aliases:
        return aliases[value]

    base = value.split("-", 1)[0]
    if base in aliases:
        return aliases[base]

    return DEFAULT_LANGUAGE


def language_instruction(language: str | None = None) -> str:
    """
    Return a system instruction restricting replies to the three languages.
    Use this when building the Ollama system prompt.
    """
    selected = normalize_language(language)
    selected_name = LANGUAGE_NAMES[selected]

    return (
        "You are a helpful local desktop assistant powered by Ollama. "
        "You may communicate ONLY in English, Telugu, or Hindi. "
        "Do not answer in any other language. "
        "Understand and respond naturally in the user's supported language. "
        f"The preferred response language is {selected_name}. "
        "If the user's language is unclear, respond in English. "
        "Keep code, commands, filenames, URLs, and technical identifiers "
        "unchanged when necessary."
    )


# ---------------------------------------------------------------------------
# Engine capabilities
# ---------------------------------------------------------------------------

DEFAULT_CAPS: dict[str, Any] = {
    "vision": False,
    "tools": False,

    # Ollama's native /api/chat format is used by the local LLM provider.
    "tool_format": "ollama",

    "max_tools": 12,
    "audio_out": False,
    "barge_in": False,
    "non_blocking": False,
    "resumption": False,

    # Language capability metadata for provider discovery and UI.
    "supported_languages": SUPPORTED_LANGUAGES,
    "default_language": DEFAULT_LANGUAGE,

    # This assistant's configured AI backend is local Ollama.
    "local_only": True,
}


# ---------------------------------------------------------------------------
# Provider configuration fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "text"
    default: Any = ""
    choices: Sequence[str] = ()
    required: bool = True
    hint: str = ""
    advanced: bool = False


# ---------------------------------------------------------------------------
# Local Ollama pipeline events
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    id: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class FunctionResponse:
    id: str = ""
    name: str = ""
    response: dict = field(default_factory=dict)
    scheduling: Any = None


@dataclass
class Msg:
    """Typed event emitted by the local PipelineEngine.

    This is a local Ollama/audio event contract. It does not imitate a
    Gemini/Live API response object.
    """

    kind: str
    text: str = ""
    audio: Optional[bytes] = None
    tool_calls: list[FunctionCall] = field(default_factory=list)

    @staticmethod
    def audio_chunk(pcm: bytes) -> "Msg":
        return Msg(kind="audio", audio=pcm)

    @staticmethod
    def said(text: str) -> "Msg":
        return Msg(kind="assistant_text", text=text)

    @staticmethod
    def heard(text: str) -> "Msg":
        return Msg(kind="user_text", text=text)

    @staticmethod
    def turn_end() -> "Msg":
        return Msg(kind="turn_end")

    @staticmethod
    def calls(items: Sequence[FunctionCall]) -> "Msg":
        return Msg(kind="tool_calls", tool_calls=list(items))


# ---------------------------------------------------------------------------
# Provider protocols
# ---------------------------------------------------------------------------

class Engine(Protocol):
    """Conversation engine consumed by main.py."""

    caps: dict

    async def send_realtime_input(self, *, audio: Any) -> None:
        """Accept microphone PCM, normally 16 kHz mono."""

    async def send_client_content(
        self,
        *,
        turns: Any,
        turn_complete: bool = True,
    ) -> None:
        """
        Accept a plain local user turn and translate it into Ollama chat
        messages. Image input is unsupported unless the selected Ollama model
        supports vision and the pipeline explicitly implements image conversion.
        """

    async def send_tool_response(
        self,
        *,
        function_responses: Any,
    ) -> None:
        """Pass local tool results back into the ongoing conversation."""

    def receive(self) -> AsyncIterator[Msg]:
        """Return an asynchronous stream of assistant events."""

    async def close(self) -> None:
        """Stop tasks and release local audio resources."""


class LLM(Protocol):
    """
    Local Ollama chat interface.

    The implementation should call Ollama's native /api/chat endpoint,
    not a hosted AI endpoint.

    Stream events:
        {"text": "partial assistant text"}
        {"tool_calls": [FunctionCall(...)]}
    """

    caps: dict

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        stream: bool = True,
    ) -> AsyncIterator[dict]:
        """
        Stream text and/or tool calls from the local Ollama model.

        The provider must add or preserve the language instruction in the
        system message and must not call an external AI service.
        """


class STT(Protocol):
    """
    Local speech-to-text interface.

    Only English, Telugu, and Hindi are allowed. A provider must detect or
    identify the recognized language and reject unsupported-language input.
    """

    caps: dict

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        language: str = "",
    ) -> str:
        """
        Return recognized text.

        `language` is a requested language code, not proof that the audio
        actually belongs to that language.
        """


class TTS(Protocol):
    """
    Local text-to-speech interface.

    The provider must use an installed local voice supporting the requested
    language. Audio chunks should be 24 kHz mono PCM if that is what the
    application's playback path expects.
    """

    caps: dict

    async def speak(
        self,
        text: str,
        voice: str = "",
    ) -> AsyncIterator[bytes]:
        """Yield local synthesized audio chunks."""


# ---------------------------------------------------------------------------
# Provider declarations
# ---------------------------------------------------------------------------

_KINDS = ("llm", "stt", "tts", "engine")
_REQUIRED_KEYS = ("id", "kind", "label", "factory")


def validate(decl: dict) -> str:
    """Return an error string if a provider declaration is invalid."""
    if not isinstance(decl, dict):
        return "PROVIDER is not a dict"

    for key in _REQUIRED_KEYS:
        if not decl.get(key):
            return f"PROVIDER is missing '{key}'"

    if decl["kind"] not in _KINDS:
        return f"kind must be one of {_KINDS}, not {decl['kind']!r}"

    if not callable(decl["factory"]):
        return "factory is not callable"

    for item in decl.get("needs", ()):
        if not isinstance(item, Field):
            return (
                "needs must contain Field objects, "
                f"found {type(item).__name__}"
            )

    return ""


def caps_of(decl: dict) -> dict:
    """Return complete capability metadata for a provider."""
    result = dict(DEFAULT_CAPS)
    result.update(decl.get("caps") or {})
    return result
