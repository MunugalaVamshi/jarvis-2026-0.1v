
"""
Ollama-only local LLM client for JARVIS.

Uses only Ollama's native HTTP API.
No OpenAI, Gemini, cloud LLM providers, or API keys.

Default URL:   http://127.0.0.1:11434
Default model: llama3.2

Configuration: config/settings.json

Optional configuration:
{
    "llm_url": "http://127.0.0.1:11434",
    "llm_model": "llama3.2"
}
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Generator

import requests


DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"

_SENT_END = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*\n")


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "settings.json"


def _load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_llm_provider() -> str:
    """Compatibility helper. The LLM backend is always Ollama."""
    return "ollama"


def get_llm_settings() -> tuple[str, str]:
    """Return the configured Ollama URL and model."""
    cfg = _load_config()

    url = str(cfg.get("ollama_url") or cfg.get("llm_url") or DEFAULT_URL).strip().rstrip("/")
    model = str(cfg.get("ollama_model") or cfg.get("llm_model") or DEFAULT_MODEL).strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError("llm_url must be an HTTP or HTTPS URL.")

    if not model:
        model = DEFAULT_MODEL

    return url, model


def _api_url(path: str) -> str:
    url, _ = get_llm_settings()
    return f"{url}/{path.lstrip('/')}"


def ensure_ollama_running(timeout: int = 15) -> bool:
    """Check Ollama and attempt to start it if it is not responding."""
    health_url = _api_url("/api/tags")

    def is_up() -> bool:
        try:
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    if is_up():
        return True

    print("[LLM] Ollama is not responding; attempting to start it.")

    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }

    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )

    try:
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        print("[LLM] Ollama was not found. Install it from https://ollama.com")
        return False
    except Exception as exc:
        print(f"[LLM] Could not start Ollama: {exc}")
        return False

    deadline = time.monotonic() + max(1, timeout)

    while time.monotonic() < deadline:
        if is_up():
            print("[LLM] Ollama is running.")
            return True
        time.sleep(0.5)

    print("[LLM] Ollama did not respond before the timeout.")
    return False


def check_model_available(log: Callable | None = None) -> bool:
    """Check whether the configured model is installed locally."""
    url, model = get_llm_settings()

    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        names = {
            item.get("name", "")
            for item in models
            if isinstance(item, dict)
        }

        # Ollama may report either "model:tag" or the configured tag.
        found = model in names or any(
            name.split(":")[0] == model.split(":")[0]
            for name in names
        )

        if not found:
            message = (
                f"Model '{model}' is not installed locally.\n"
                f"Available models: {', '.join(sorted(names)) or 'none'}\n"
                f"Install it with: ollama pull {model}"
            )
            print(f"[LLM] {message}")
            if log:
                log(message)

        return found

    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"[LLM] Could not check model availability: {exc}")
        return False


def warmup_model(system_prompt: str | None = None) -> bool:
    """Load the configured model with a minimal local request."""
    _, model = get_llm_settings()

    if not ensure_ollama_running():
        return False

    if not check_model_available():
        return False

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    messages.append({"role": "user", "content": "hi"})

    try:
        response = requests.post(
            _api_url("/api/chat"),
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 1},
            },
            timeout=180,
        )
        response.raise_for_status()
        print(f"[LLM] Ollama model '{model}' is ready.")
        return True

    except requests.RequestException as exc:
        print(f"[LLM] Ollama warmup failed: {exc}")
        return False


def _post_chat(
    messages: list,
    model: str,
    timeout: int,
    tools: list | None = None,
    stream: bool = False,
    num_predict: int = 150,
):
    """Send a request exclusively to Ollama's native chat endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": -1,
        "options": {"num_predict": num_predict},
    }

    if tools:
        payload["tools"] = tools

    return requests.post(
        _api_url("/api/chat"),
        json=payload,
        timeout=timeout,
        stream=stream,
    )


def _parse_response(response) -> dict:
    """Normalize an Ollama chat response."""
    data = response.json()
    message = data.get("message") or {}

    return {
        "content": (message.get("content") or "").strip(),
        "tool_calls": message.get("tool_calls") or [],
    }


def _request_chat(
    messages: list,
    model: str,
    timeout: int,
    tools: list | None = None,
    stream: bool = False,
    num_predict: int = 150,
):
    """Make one Ollama request and report useful local connection errors."""
    try:
        response = _post_chat(
            messages=messages,
            model=model,
            timeout=timeout,
            tools=tools,
            stream=stream,
            num_predict=num_predict,
        )
        response.raise_for_status()
        return response

    except requests.exceptions.ConnectionError as exc:
        url, _ = get_llm_settings()
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Ensure Ollama is installed and running."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after {timeout} seconds."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None else "unknown"
        )
        detail = (
            exc.response.text[:300]
            if exc.response is not None else str(exc)
        )
        raise RuntimeError(
            f"Ollama HTTP error {status}: {detail}"
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def call_llm(
    messages: list,
    tools: list | None = None,
    timeout: int = 120,
) -> dict:
    """
    Non-streaming Ollama chat.

    Returns:
        {"content": str, "tool_calls": list}
    """
    _, model = get_llm_settings()

    response = _request_chat(
        messages=messages,
        model=model,
        timeout=timeout,
        tools=tools,
    )

    try:
        return _parse_response(response)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Could not parse Ollama response: {exc}"
        ) from exc


def call_llm_text(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """Generate plain text using the local Ollama model."""
    _, default_model = get_llm_settings()
    selected_model = model or default_model

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system,
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    response = _request_chat(
        messages=messages,
        model=selected_model,
        timeout=timeout,
        num_predict=600,
    )

    try:
        return (
            response.json()
            .get("message", {})
            .get("content") or ""
        ).strip()
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Could not parse Ollama text response: {exc}"
        ) from exc


def _extract_sentence(buffer: str) -> tuple[str | None, str]:
    """Extract a complete sentence from a response buffer."""
    match = _SENT_END.search(buffer)

    if not match:
        return None, buffer

    sentence = buffer[:match.start()].strip()
    remaining = buffer[match.end():]

    return sentence or None, remaining


def call_llm_stream(
    messages: list,
    tools: list | None = None,
    timeout: int = 120,
) -> Generator[dict, None, None]:
    """
    Stream output from Ollama.

    Yields:
        {"type": "sentence", "text": str}
        {"type": "done", "content": str, "tool_calls": list}
    """
    _, model = get_llm_settings()

    response = _request_chat(
        messages=messages,
        model=model,
        timeout=timeout,
        tools=tools,
        stream=True,
        num_predict=150,
    )

    full_content = ""
    buffer = ""
    tool_calls = []

    try:
        with response:
            for raw in response.iter_lines():
                if not raw:
                    continue

                try:
                    chunk = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                message = chunk.get("message") or {}
                delta = message.get("content") or ""

                full_content += delta
                buffer += delta

                while True:
                    sentence, buffer = _extract_sentence(buffer)

                    if sentence is None:
                        break

                    yield {
                        "type": "sentence",
                        "text": sentence,
                    }

                tool_calls.extend(message.get("tool_calls") or [])

                if chunk.get("error"):
                    raise RuntimeError(
                        f"Ollama stream error: {chunk['error']}"
                    )

                if chunk.get("done"):
                    if buffer.strip():
                        yield {
                            "type": "sentence",
                            "text": buffer.strip(),
                        }

                    yield {
                        "type": "done",
                        "content": full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

        # Handle servers that close the stream without a done marker.
        if buffer.strip():
            yield {
                "type": "sentence",
                "text": buffer.strip(),
            }

        yield {
            "type": "done",
            "content": full_content.strip(),
            "tool_calls": tool_calls,
        }

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama streaming request failed: {exc}"
        ) from exc
