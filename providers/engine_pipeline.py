
"""
providers/engine_pipeline.py

Local voice conversation pipeline:
    Microphone -> local Whisper -> local Ollama -> local TTS -> speaker

Supported platforms:
    Windows and Linux only.

No OpenAI, Gemini, Groq, ElevenLabs, or other hosted AI APIs are used here.

Required provider contract:
    llm.chat(messages, tools) -> async iterator of dictionaries containing
        {"text": "..."}
        {"tool_calls": [FunctionCall, ...]}

    stt.transcribe(pcm, sample_rate) -> awaitable string
    tts.speak(text, language=...) -> async iterator of PCM bytes

The Ollama provider must execute against its local Ollama server.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import sys
import time
from typing import Any, AsyncIterator, Optional
from . import schema
from .base import DEFAULT_CAPS, FunctionCall, Msg


_DEFAULT_HISTORY_CHARS = 24_000

_VAD_SILENCE_MS = 700
_VAD_MIN_SPEECH_MS = 250
_VAD_MAX_TURN_S = 30.0
_VAD_FLOOR_MARGIN = 3.5
_VAD_ABSOLUTE_MIN = 0.006

_MAX_TOOL_ROUNDS = 4


def _check_supported_platform() -> None:
    """Allow Windows and Linux only."""
    if not (
        sys.platform.startswith("win")
        or sys.platform.startswith("linux")
    ):
        raise RuntimeError(
            f"Unsupported platform: {sys.platform}. "
            "This assistant supports Windows and Linux only."
        )


class PipelineEngine:
    """Local STT -> Ollama -> local TTS engine."""

    def __init__(
        self,
        llm,
        stt=None,
        tts=None,
        *,
        config=None,
        log=None,
        history_chars: int = _DEFAULT_HISTORY_CHARS,
        wake_word_required: bool = False,
    ):
        _check_supported_platform()

        if llm is None:
            raise RuntimeError(
                "The pipeline requires the local Ollama provider."
            )

        self._llm = llm
        self._stt = stt
        self._tts = tts
        self._log = log or (lambda message: None)
        self._history_chars = history_chars
        self.wake_word_required = bool(wake_word_required)

        self.caps = dict(DEFAULT_CAPS)

        llm_caps = getattr(llm, "caps", {}) or {}
        self.caps.update({
            "tools": bool(llm_caps.get("tools")),
            "tool_format": llm_caps.get("tool_format"),
            "max_tools": llm_caps.get("max_tools"),
            "max_tool_chars": llm_caps.get("max_tool_chars"),
            "vision": bool(llm_caps.get("vision")),
            "audio_out": tts is not None,
            "barge_in": False,
            "non_blocking": False,
            "resumption": False,
        })

        self._system = ""
        self._tools: Optional[list] = None
        self._declarations: list = []

        if config is not None:
            self._read_config(config)

        self._history: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._gen: Optional[asyncio.Task] = None
        self._closed = False

        self._awaiting: dict[str, str] = {}
        self._done_calls: dict[str, str] = {}
        self._pending_sigs: dict[str, str] = {}
        self._rounds = 0
        self._followup = False

        self._mic: list[bytes] = []
        self._mic_bytes = 0

        self.auto_turn = False
        self._on_turn_end = None

        self._vad_floor = _VAD_ABSOLUTE_MIN
        self._vad_speech_ms = 0.0
        self._vad_quiet_ms = 0.0
        self._vad_started = False

        self._timing: dict = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _read_config(self, config) -> None:
        if isinstance(config, dict):
            system_instruction = config.get("system_instruction")
            tool_groups = config.get("tools") or []
        else:
            system_instruction = getattr(
                config, "system_instruction", None
            )
            tool_groups = getattr(config, "tools", None) or []

        if system_instruction:
            self._system = (
                system_instruction
                if isinstance(system_instruction, str)
                else str(system_instruction)
            )

        declarations = []

        for group in tool_groups:
            if isinstance(group, dict):
                functions = group.get("function_declarations")
            else:
                functions = getattr(
                    group, "function_declarations", None
                )

            if functions:
                declarations.extend(functions)

        self._declarations = schema.budget(
            declarations,
            max_tools=self.caps.get("max_tools"),
            max_chars=self.caps.get("max_tool_chars"),
        )

        tool_format = self.caps.get("tool_format")

        if not self.caps.get("tools") or not self._declarations:
            self._tools = None

        elif tool_format == "prompt":
            self._system = (
                self._system
                + "\n\n"
                + schema.to_prompt(self._declarations)
            ).strip()
            self._tools = None

        else:
            # schema.convert must produce the tool format expected by
            # the local Ollama provider.
            self._tools = schema.convert(
                self._declarations,
                tool_format,
            )

        if len(self._declarations) < len(declarations):
            self._log(
                f"SYS: declared {len(self._declarations)} of "
                f"{len(declarations)} available tools for this model."
            )

    def tool_names(self) -> list[str]:
        names = []

        for declaration in self._declarations:
            if isinstance(declaration, dict):
                name = declaration.get("name", "")
            else:
                name = getattr(declaration, "name", "")

            if name:
                names.append(name)

        return names

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def _messages(self) -> list[dict]:
        messages = []

        if self._system:
            messages.append({
                "role": "system",
                "content": self._system,
            })

        messages.extend(self._trimmed())
        return messages

    def _trimmed(self) -> list[dict]:
        total = 0
        cut = 0

        for index in range(len(self._history) - 1, -1, -1):
            total += len(json.dumps(
                self._history[index],
                ensure_ascii=False,
                default=str,
            ))

            if total > self._history_chars:
                cut = index + 1
                break

        # Avoid retaining tool results without their preceding exchange.
        while (
            cut < len(self._history)
            and self._history[cut].get("role") == "tool"
        ):
            cut += 1

        return self._history[cut:]

    # ------------------------------------------------------------------
    # Microphone input
    # ------------------------------------------------------------------

    async def send_realtime_input(self, *, audio) -> None:
        """Buffer microphone PCM until push-to-talk or VAD ends the turn."""
        if self._closed or self._stt is None:
            return

        data = _pcm_of(audio)

        if not data:
            return

        self._mic.append(data)
        self._mic_bytes += len(data)

        if self.auto_turn:
            await self._watch_for_turn_end(data)

    async def _watch_for_turn_end(self, frame: bytes) -> None:
        from . import audio as audio_utils

        samples = audio_utils.to_float32(frame)

        if samples.size == 0:
            return

        import numpy as np

        rms = float(np.sqrt(np.mean(samples * samples)))
        milliseconds = (
            samples.size / audio_utils.MIC_RATE * 1000.0
        )

        if rms < self._vad_floor:
            self._vad_floor = rms
        else:
            self._vad_floor += (
                rms - self._vad_floor
            ) * 0.002

        threshold = max(
            self._vad_floor * _VAD_FLOOR_MARGIN,
            _VAD_ABSOLUTE_MIN,
        )

        if rms >= threshold:
            self._vad_speech_ms += milliseconds
            self._vad_quiet_ms = 0.0

            if self._vad_speech_ms >= _VAD_MIN_SPEECH_MS:
                self._vad_started = True

            return

        if not self._vad_started:
            keep_bytes = (
                int(audio_utils.MIC_RATE * 0.3)
                * audio_utils.SAMPLE_WIDTH
            )

            while (
                self._mic_bytes > keep_bytes
                and len(self._mic) > 1
            ):
                self._mic_bytes -= len(self._mic.pop(0))

            self._vad_speech_ms = 0.0
            return

        self._vad_quiet_ms += milliseconds

        max_bytes = (
            int(audio_utils.MIC_RATE * _VAD_MAX_TURN_S)
            * audio_utils.SAMPLE_WIDTH
        )

        if (
            self._vad_quiet_ms >= _VAD_SILENCE_MS
            or self._mic_bytes >= max_bytes
        ):
            self._reset_vad()
            await self.flush_microphone()

    def _reset_vad(self) -> None:
        self._vad_speech_ms = 0.0
        self._vad_quiet_ms = 0.0
        self._vad_started = False

    async def flush_microphone(
        self,
        sample_rate: int = 16000,
    ) -> str:
        """Run local speech recognition and submit the recognized text."""
        if self._closed or self._stt is None or not self._mic:
            return ""

        pcm = b"".join(self._mic)
        self._mic.clear()
        self._mic_bytes = 0

        from . import audio as audio_utils

        self._timing = {
            "heard_s": audio_utils.duration_s(pcm, sample_rate)
        }

        started = time.monotonic()

        try:
            text = (
                await self._stt.transcribe(pcm, sample_rate)
            ).strip()

        except Exception as exc:
            self._report("local speech recognition", self._stt, exc)
            return ""

        self._timing["stt"] = time.monotonic() - started

        if text and self.wake_word_required:
            # Detect the wake phrase in the same Whisper transcript as the
            # command. A separate wake detector can consume the audio frame
            # containing the command immediately after "Hey Jarvis".
            match = re.match(
                r"^\s*(?:hey[\s,.-]*)?jarvis\b[\s,.:;!?-]*",
                text,
                flags=re.IGNORECASE,
            )
            if not match:
                return ""
            text = text[match.end():].strip()
            if not text:
                await self._speak("Yes?")
                return ""

        if text:
            await self._emit(Msg.heard(text))
            await self.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]}
            )

        return text

    # ------------------------------------------------------------------
    # Text and image input
    # ------------------------------------------------------------------

    async def send_client_content(
        self,
        *,
        turns,
        turn_complete: bool = True,
    ) -> None:
        """Add user text or supported image content to the conversation."""
        if self._closed:
            return

        parts = (
            turns.get("parts", [])
            if isinstance(turns, dict)
            else getattr(turns, "parts", [])
        )

        content = self._to_content(parts or [])

        if content == "" or content == []:
            return

        self._history.append({
            "role": "user",
            "content": content,
        })

        self._done_calls.clear()
        self._rounds = 0

        self._queue_generation()

    def _to_content(self, parts: list) -> Any:
        """Convert text parts without importing an OpenAI adapter."""
        text_parts = []
        images = []

        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                inline = (
                    part.get("inline_data")
                    or part.get("inlineData")
                )
            else:
                text = getattr(part, "text", None)
                inline = getattr(part, "inline_data", None)

            if isinstance(text, str) and text.strip():
                text_parts.append(text)

            if inline:
                if isinstance(inline, dict):
                    image_data = inline.get("data")
                else:
                    image_data = getattr(inline, "data", None)

                if image_data:
                    images.append(image_data)

        content = "\n".join(text_parts).strip()

        if not images:
            return content

        if not self.caps.get("vision"):
            return (
                content
                + "\n[An image was attached, but the configured "
                  "Ollama model does not support vision.]"
            ).strip()

        # The local Ollama adapter must translate this representation
        # into Ollama's native message content/images fields.
        return {
            "text": content,
            "images": images,
        }

    # ------------------------------------------------------------------
    # Tool responses and event queue
    # ------------------------------------------------------------------

    async def send_tool_response(self, *, function_responses) -> None:
        """Record tool results without blocking the receive event loop."""
        if self._closed:
            return

        started = self._timing.get("tool_at")

        if started:
            self._tick("tools", time.monotonic() - started)
            self._timing["tool_names"] = ", ".join(
                sorted(set(self._awaiting.values()))
            )

        for response in function_responses or []:
            call_id = str(getattr(response, "id", "") or "")
            name = str(
                getattr(response, "name", "")
                or self._awaiting.get(call_id, "")
            )

            payload = getattr(response, "response", None)

            if isinstance(payload, dict) and "result" in payload:
                payload = payload["result"]

            signature = self._pending_sigs.pop(call_id, "")

            if signature:
                self._done_calls[signature] = _stringify(payload)

            self._history.append({
                "role": "tool",
                "tool_call_id": call_id or name,
                "name": name,
                "content": _stringify(payload),
            })

            self._awaiting.pop(call_id, None)

        if not self._awaiting:
            self._start_generation()

    async def receive(self) -> AsyncIterator[Msg]:
        while True:
            message = await self._queue.get()

            if message is None:
                return

            yield message

    async def close(self) -> None:
        self._closed = True

        if self._gen and not self._gen.done():
            self._gen.cancel()

            try:
                await self._gen
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        await self._queue.put(None)

    async def warmup(self) -> None:
        """Warm up the local speech recognition and speech engines."""
        jobs = []

        for component in (self._stt, self._tts):
            warmup = getattr(component, "warmup", None)

            if warmup is not None:
                try:
                    result = warmup(self._log)
                except TypeError:
                    result = warmup()

                if inspect.isawaitable(result):
                    jobs.append(result)

        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

        # Optionally warm up Ollama if the provider exposes warmup().
        warmup = getattr(self._llm, "warmup", None)

        if warmup is not None:
            try:
                result = warmup()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._report("local Ollama warmup", self._llm, exc)

    # ------------------------------------------------------------------
    # Interruption and generation scheduling
    # ------------------------------------------------------------------

    def _queue_generation(self) -> None:
        if self._closed:
            return

        if self._gen and not self._gen.done():
            self._followup = True
            return

        self._gen = asyncio.create_task(self._generate())

    def _start_generation(self) -> None:
        if self._closed:
            return

        if self._gen and not self._gen.done():
            self._gen.cancel()

        self._gen = asyncio.create_task(self._generate())

    # ------------------------------------------------------------------
    # Ollama generation and tool execution requests
    # ------------------------------------------------------------------

    async def _generate(self) -> None:
        said = []
        calls = []
        started = time.monotonic()
        first_word = None
        reasoning = _ReasoningFilter()
        pending = ""

        try:
            # This interface must be implemented by providers/llm_ollama.py.
            # It must stream native Ollama /api/chat results as event dicts.
            async for event in self._llm.chat(
                self._messages(),
                self._tools,
            ):
                if not isinstance(event, dict):
                    continue

                if event.get("error"):
                    raise RuntimeError(str(event["error"]))

                chunk = event.get("text") or ""

                if chunk:
                    visible = reasoning.feed(chunk)

                    if visible and first_word is None:
                        first_word = time.monotonic() - started

                    pending += visible
                    sentences, pending = _take_sentences(pending)

                    for sentence in sentences:
                        said.append(sentence)
                        await self._emit(Msg.said(sentence))
                        await self._speak(sentence)

                event_calls = event.get("tool_calls")

                if event_calls:
                    calls.extend(
                        _normalise_calls(event_calls)
                    )

            pending += reasoning.flush()

            if pending.strip():
                final_text = pending.strip()
                said.append(final_text)

                if first_word is None:
                    first_word = time.monotonic() - started

                await self._emit(Msg.said(final_text))
                await self._speak(final_text)

            if reasoning.hid:
                self._timing["thought"] = reasoning.hid

            text = "".join(said)

            # Models without native tool calling may return a JSON tool
            # request when schema.to_prompt() was used.
            if (
                not calls
                and self.caps.get("tool_format") == "prompt"
            ):
                parsed = _parse_prompt_call(
                    text,
                    set(self.tool_names()),
                )

                if parsed:
                    calls = [parsed]
                    text = ""

            if calls:
                self._rounds += 1

                fresh = []
                repeated = []

                for call in calls:
                    signature = _signature(call)

                    if signature in self._done_calls:
                        repeated.append(call)
                    else:
                        fresh.append(call)

                if repeated:
                    self._log(
                        "SYS: Ollama repeated a completed tool call; "
                        "reusing its previous result."
                    )

                    for call in repeated:
                        result = self._done_calls[
                            _signature(call)
                        ]

                        self._history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.args,
                                },
                            }],
                        })

                        self._history.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": result,
                        })

                if self._rounds > _MAX_TOOL_ROUNDS:
                    self._log(
                        "SYS: maximum tool rounds reached; "
                        "asking Ollama to answer without more tools."
                    )

                    self._history.append({
                        "role": "user",
                        "content": (
                            "Do not call any more tools. Answer now, "
                            "using the results already available."
                        ),
                    })

                    fresh = []

                if not fresh:
                    self._start_generation()
                    return

                for call in fresh:
                    self._done_calls.setdefault(
                        _signature(call),
                        "",
                    )

                self._history.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.args,
                            },
                        }
                        for call in fresh
                    ],
                })

                self._awaiting = {
                    call.id: call.name for call in fresh
                }

                self._pending_sigs = {
                    call.id: _signature(call)
                    for call in fresh
                }

                self._tick(
                    "llm",
                    time.monotonic() - started,
                    first_word,
                    text,
                )

                self._timing["tool_at"] = time.monotonic()

                # main.py receives these and executes the real actions.
                # Do not wait for tool results here.
                await self._emit(Msg.calls(fresh))
                return

            self._tick(
                "llm",
                time.monotonic() - started,
                first_word,
                text,
            )

            if text:
                self._history.append({
                    "role": "assistant",
                    "content": text,
                })

            self._log(
                f"SYS: local Ollama replied in "
                f"{time.monotonic() - started:.1f}s"
            )

            self._print_timing()
            await self._emit(Msg.turn_end())

            if self._followup:
                self._followup = False
                self._gen = asyncio.create_task(self._generate())

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self._report("local Ollama", self._llm, exc)

            await self._emit(
                Msg.said(_failure_line("local Ollama", exc))
            )
            await self._emit(Msg.turn_end())

    # ------------------------------------------------------------------
    # Local speech synthesis
    # ------------------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if self._tts is None:
            return

        text = _speakable(text)

        if not text:
            return

        from . import audio as audio_utils

        started = time.monotonic()
        first_audio = None
        total_bytes = 0

        kwargs = {}
        language = getattr(
            self._stt,
            "detected_language",
            "",
        )

        if language and _accepts_language(self._tts):
            kwargs["language"] = language

        try:
            async for pcm in self._tts.speak(text, **kwargs):
                if not pcm:
                    continue

                if first_audio is None:
                    first_audio = time.monotonic() - started

                total_bytes += len(pcm)
                await self._emit(Msg.audio_chunk(pcm))

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self._report("local speech synthesis", self._tts, exc)
            return

        timing = self._timing
        timing["tts"] = timing.get("tts", 0.0) + (
            time.monotonic() - started
        )

        if timing.get("tts_first") is None:
            timing["tts_first"] = first_audio

        timing["spoke_s"] = timing.get("spoke_s", 0.0) + (
            total_bytes
            / float(
                audio_utils.SPEAKER_RATE
                * audio_utils.SAMPLE_WIDTH
            )
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _tick(
        self,
        key: str,
        value,
        first=None,
        text: str = "",
    ) -> None:
        timing = self._timing
        timing[key] = timing.get(key, 0.0) + float(value or 0.0)

        if first is not None and timing.get(f"{key}_first") is None:
            timing[f"{key}_first"] = first

        if text:
            timing["chars"] = timing.get("chars", 0) + len(text)

    def _print_timing(self) -> None:
        if os.environ.get("MARK_NO_TIMING"):
            self._timing = {}
            return

        timing, self._timing = self._timing, {}

        if not timing:
            return

        def row(label, seconds, detail):
            bar = "#" * min(30, int((seconds or 0) * 2))
            return (
                f"[Pipeline] {label:<7}"
                f"{(seconds or 0):>7.2f}s  "
                f"{bar:<30}  {detail}"
            )

        lines = ["[Pipeline] -- local turn " + "-" * 43]
        total = 0.0

        if "stt" in timing:
            total += timing["stt"]
            lines.append(row(
                "hear",
                timing["stt"],
                f"{timing.get('heard_s', 0):.1f}s of audio | "
                f"{_name_of(self._stt)}",
            ))

        if "llm" in timing:
            total += timing["llm"]
            first = timing.get("llm_first")
            detail = (
                f"first text {first:.2f}s | "
                if first is not None
                else ""
            )

            lines.append(row(
                "think",
                timing["llm"],
                f"{detail}{timing.get('chars', 0)} chars | "
                f"{_name_of(self._llm)}",
            ))

        if "tools" in timing:
            total += timing["tools"]
            lines.append(row(
                "tools",
                timing["tools"],
                timing.get("tool_names", ""),
            ))

        if "tts" in timing:
            total += timing["tts"]
            first = timing.get("tts_first")
            detail = (
                f"first audio {first:.2f}s | "
                if first is not None
                else ""
            )

            lines.append(row(
                "speak",
                timing["tts"],
                f"{detail}{timing.get('spoke_s', 0):.1f}s audio | "
                f"{_name_of(self._tts)}",
            ))

        lines.append("[Pipeline] " + "-" * 67)
        lines.append(f"[Pipeline] {'total':<7}{total:>7.2f}s")

        _console("\n".join(lines))

    def _report(self, component: str, provider, exc: Exception) -> None:
        import traceback

        target = _target_of(provider)
        detail = " ".join(str(exc).split())[:200]
        if not detail:
            detail = exc.__class__.__name__

        suffix = f" at {target}" if target else ""

        self._log(
            f"SYS: {component} failed{suffix}: "
            f"{type(exc).__name__}: {detail}"
        )

        _console(
            f"[Pipeline] {component} failed{suffix}: "
            f"{type(exc).__name__}: {detail}"
        )

        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )

    async def _emit(self, message: Msg) -> None:
        if not self._closed:
            await self._queue.put(message)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _ReasoningFilter:
    """Hide streamed <think>, <thinking>, and <reasoning> sections."""

    _OPEN = ("<think>", "<thinking>", "<reasoning>")
    _CLOSE = ("</think>", "</thinking>", "</reasoning>")
    _HOLD = max(len(tag) for tag in _OPEN + _CLOSE)

    def __init__(self):
        self._buf = ""
        self._inside = False
        self.hid = 0

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        output = []
        keep = self._HOLD - 1

        while True:
            if self._inside:
                index, tag = _first_of(self._buf, self._CLOSE)

                if index < 0:
                    if len(self._buf) > keep:
                        self.hid += len(self._buf) - keep
                        self._buf = self._buf[-keep:]
                    return "".join(output)

                self.hid += index + len(tag)
                self._buf = self._buf[index + len(tag):]
                self._inside = False
                continue

            index, tag = _first_of(self._buf, self._OPEN)

            if index < 0:
                if len(self._buf) > keep:
                    output.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                return "".join(output)

            output.append(self._buf[:index])
            self._buf = self._buf[index + len(tag):]
            self._inside = True

    def flush(self) -> str:
        if self._inside:
            self.hid += len(self._buf)
            self._buf = ""
            return ""

        result = self._buf
        self._buf = ""
        return result


def _first_of(text: str, tags) -> tuple[int, str]:
    best_index = -1
    best_tag = ""

    for tag in tags:
        index = text.find(tag)

        if index >= 0 and (
            best_index < 0 or index < best_index
        ):
            best_index = index
            best_tag = tag

    return best_index, best_tag


_SENTENCE_END = re.compile(
    r"(?<=[.!?…])\s+|(?<=[。！？؟۔।])\s*"
)

_UNSPEAKABLE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001FC00-\U0001FFFD"
    "\u2600-\u27BF"
    "\u2190-\u21FF"
    "\u2300-\u23FF"
    "\uFE00-\uFE0F"
    "\u200D"
    "\u2122\u2139\u2049\u203C"
    "]+",
    flags=re.UNICODE,
)


def _signature(call) -> str:
    try:
        arguments = json.dumps(
            call.args or {},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        arguments = str(getattr(call, "args", ""))

    return f"{call.name}({arguments})"


def _speakable(text: str) -> str:
    cleaned = _UNSPEAKABLE.sub(" ", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _take_sentences(buffer: str) -> tuple[list[str], str]:
    sentences = []

    while True:
        match = _SENTENCE_END.search(buffer)

        if not match:
            break

        sentence = buffer[:match.end()].strip()

        if sentence:
            sentences.append(sentence)

        buffer = buffer[match.end():]

    if len(buffer) > 320:
        cut = buffer.rfind(" ", 0, 320)

        if cut > 0:
            sentences.append(buffer[:cut].strip())
            buffer = buffer[cut:].lstrip()

    return sentences, buffer


def _accepts_language(tts) -> bool:
    try:
        return "language" in inspect.signature(
            tts.speak
        ).parameters
    except Exception:
        return False


def _console(text: str) -> None:
    try:
        print(text)
    except Exception:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "replace").decode(encoding, "replace"))
        except Exception:
            pass


def _name_of(provider) -> str:
    if provider is None:
        return "none"

    for attribute in (
        "last_voice",
        "model",
        "model_name",
        "voice",
    ):
        value = getattr(provider, attribute, "")

        if value:
            device = getattr(provider, "device", "")
            return f"{value}/{device}" if device else str(value)

    engine = getattr(provider, "_engine", "")
    return str(engine) if engine else provider.__class__.__name__


def _target_of(provider) -> str:
    if provider is None:
        return ""

    for attribute in ("base", "base_url", "endpoint"):
        value = getattr(provider, attribute, "")

        if value:
            return str(value)

    return provider.__class__.__name__


def _failure_line(component: str, exc: Exception) -> str:
    name = type(exc).__name__.lower()

    if "connect" in name or "timeout" in name:
        return (
            f"I could not reach {component}. "
            "Check that the local service is running and its "
            "configuration is correct. Details are in the log."
        )

    return (
        f"Something went wrong with {component}. "
        "The details are in the log."
    )


def _pcm_of(audio: Any) -> bytes:
    if audio is None:
        return b""

    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    if isinstance(audio, dict):
        data = audio.get("data")
    else:
        data = getattr(audio, "data", None)

    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)

    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            return b""

    return b""


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value

    if value is None:
        return "ok"

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def _normalise_calls(items) -> list[FunctionCall]:
    """Accept FunctionCall instances or native Ollama tool-call dictionaries."""
    calls = []

    for item in items or []:
        if isinstance(item, FunctionCall):
            calls.append(item)
            continue

        if not isinstance(item, dict):
            continue

        function = item.get("function") or item
        name = function.get("name")

        if not name:
            continue

        arguments = function.get("arguments") or {}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        call_id = str(item.get("id") or f"call_{name}")

        calls.append(FunctionCall(
            id=call_id,
            name=str(name),
            args=arguments,
        ))

    return calls


def _parse_prompt_call(
    text: str,
    known: set[str],
) -> Optional[FunctionCall]:
    if not text or "{" not in text:
        return None

    chunk = text[text.find("{"):text.rfind("}") + 1]

    try:
        value = json.loads(chunk)
    except Exception:
        return None

    if not isinstance(value, dict):
        return None

    name = str(
        value.get("tool") or value.get("name") or ""
    ).strip()

    if name not in known:
        return None

    arguments = value.get("args")

    if not isinstance(arguments, dict):
        arguments = {}

    return FunctionCall(
        id=f"call_{name}",
        name=name,
        args=arguments,
    )


