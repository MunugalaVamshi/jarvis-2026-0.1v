import json
import sys
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"

def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def config_exists() -> bool:
    return CONFIG_FILE.exists()

def get_assistant_name() -> str:
    """Return the configured assistant name, or 'JARVIS' if not set."""
    return load_settings().get("assistant_name", "JARVIS") or "JARVIS"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_settings().get("user_name", "MV") or "MV"


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["assistant_name"] = assistant_name.strip() or "JARVIS"
    data["user_name"] = user_name.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ── Assistant voice ──────────────────────────────────────────────────────────
# Local speech preference. System TTS uses the computer's installed voice.
AVAILABLE_VOICES = ["default"]
DEFAULT_VOICE    = "default"


def get_voice() -> str:
    """Return the configured Live voice, falling back to the default if unset
    or if the stored value is not a voice we recognise."""
    v = load_settings().get("voice_name", DEFAULT_VOICE) or DEFAULT_VOICE
    return v if v in AVAILABLE_VOICES else DEFAULT_VOICE


def save_voice(voice_name: str) -> None:
    """Persist the chosen Live voice. Unknown names collapse to the default so a
    bad value can never reach the API and break the session."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    v = (voice_name or "").strip()
    data["voice_name"] = v if v in AVAILABLE_VOICES else DEFAULT_VOICE
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_wake_word_enabled() -> bool:
    """Whether local wake-word gating is on (assistant sleeps until 'Jarvis')."""
    return load_settings().get("wake_word_enabled", True)


def save_wake_word_enabled(enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["wake_word_enabled"] = bool(enabled)
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_push_to_talk_enabled() -> bool:
    """Hold-a-key-to-speak. When on, the mic is closed unless the chord is held."""
    return load_settings().get("push_to_talk_enabled", True)


def save_push_to_talk_enabled(enabled: bool) -> None:
    _save_flag("push_to_talk_enabled", enabled)


HUD_STYLES = ("face", "core")


def get_hud_style() -> str:
    """Which centrepiece the HUD draws: the animated head, or the reactor core.

    Taste, not capability — both render in the same software painter and cost
    about the same. Defaults to the head because that is what MV shipped
    with; anyone who preferred the older look can switch back in ⚙ and the
    choice survives a restart.
    """
    v = str(load_settings().get("hud_style", "face")).strip().lower()
    return v if v in HUD_STYLES else "face"


def save_hud_style(style: str) -> None:
    s = str(style or "").strip().lower()
    _save_flag("hud_style", s if s in HUD_STYLES else "face")


# ── Live-session tuning ──────────────────────────────────────────────────────
# Everything here is optional and has a working default, so an untouched
# config behaves exactly like a configured one. Each value is also a way out:
# if a future model dislikes one of these, set it back and nothing else changes.

def get_thinking_enabled() -> bool:
    """Whether the Live model may spend tokens thinking before it answers.

    Off by default. A voice assistant is judged on how fast it starts talking,
    and the reasoning that actually needs deliberation in this app is delegated
    to the planning tools, which run on a separate non-Live model.
    """
    return bool(load_settings().get("thinking_enabled", False))


def save_thinking_enabled(enabled: bool) -> None:
    _save_flag("thinking_enabled", enabled)


def get_turn_tuning() -> dict:
    """How eagerly the server decides you have stopped speaking.

    OFF by default, and that default was earned. Cutting turns shorter looks
    like a free speed win and is not: proactive audio has to judge whether an
    utterance was even addressed to the assistant, and a turn clipped early
    gives it less to judge, so it stays quiet — and the reply to your first
    sentence only arrives once your second one has given it enough context.
    That reads as the assistant being a turn behind, which is far worse than
    the fraction of a second the tuning saves.

    Turn it on with "turn_tuning": {"enabled": true} if your own microphone and
    speaking pace suit it. `silence_ms` is the one that is felt: the pause the
    server sits through before accepting your turn is over.
    """
    cfg = load_settings().get("turn_tuning")
    cfg = cfg if isinstance(cfg, dict) else {}

    def _int(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(cfg.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "enabled":    bool(cfg.get("enabled", False)),
        "silence_ms": _int("silence_ms", 550, 200, 3000),
        "prefix_ms":  _int("prefix_ms", 150, 0, 1000),
        # "high" = quicker to decide speech has ended.
        "end_sensitivity":   str(cfg.get("end_sensitivity", "high")).lower(),
        "start_sensitivity": str(cfg.get("start_sensitivity", "default")).lower(),
    }


def save_turn_tuning(values: dict) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    cur = data.get("turn_tuning")
    cur = dict(cur) if isinstance(cur, dict) else {}
    cur.update(values or {})
    data["turn_tuning"] = cur
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")



MEDIA_RESOLUTIONS = ("default", "low", "medium", "high")


def get_media_resolution() -> str:
    """How finely the model tokenises the screenshots and camera frames it is
    sent. 'medium' keeps on-screen text readable at a fraction of the tokens a
    full-resolution frame costs; 'low' is cheaper still but starts losing small
    text, which is most of what screen captures are for."""
    v = str(load_settings().get("media_resolution", "medium")).strip().lower()
    return v if v in MEDIA_RESOLUTIONS else "medium"


def save_media_resolution(value: str) -> None:
    v = str(value or "").strip().lower()
    _save_flag("media_resolution", v if v in MEDIA_RESOLUTIONS else "medium")


def _save_flag(key: str, value) -> None:
    """Read-modify-write one key without disturbing the rest of the config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[key] = bool(value) if isinstance(value, bool) else value
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_brief_enabled() -> bool:
    return load_settings().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["morning_brief_enabled"] = enabled
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ── Audio devices ────────────────────────────────────────────────────────────
# Stored as device NAMES, not sounddevice indices. Indices shift every time a
# USB device is plugged in or removed, so a saved index silently starts pointing
# at a different microphone. The empty string means "system default", which is
# both the factory setting and what an unresolvable saved device falls back to —
# so unplugging a headset degrades to the built-in speakers instead of crashing.

def _patch_config(**fields) -> None:
    """Read-modify-write one or more keys in settings.json.

    Every setter in this file open-coded this. Collapsing it here means a new
    setting is one line, and there is one place where a corrupt config file is
    handled instead of nine."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(fields)
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_input_device() -> str:
    """Microphone device name, or '' for the system default."""
    return (load_settings().get("input_device", "") or "").strip()


def save_input_device(name: str) -> None:
    _patch_config(input_device=(name or "").strip())


def get_output_device() -> str:
    """Speaker device name, or '' for the system default."""
    return (load_settings().get("output_device", "") or "").strip()


def save_output_device(name: str) -> None:
    _patch_config(output_device=(name or "").strip())


"""
core/config_manager.py
JARVIS configuration manager — Ollama-only local LLM setup.

LLM: Ollama
Default Ollama URL: http://localhost:11434
Default model: llama3.2
STT: local Whisper
TTS: Windows system speech or local Piper

This module stores settings only. It does not call a hosted AI API.
"""

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CONFIG_DIR = BASE_DIR / "config"

# Keep the existing filename to avoid breaking existing installations.
CONFIG_FILE = CONFIG_DIR / "settings.json"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Read/write helpers
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    """
    Load the existing configuration file.

    The function name is retained for compatibility with existing imports.
    This configuration no longer stores or requires hosted AI API keys.
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to load configuration: {exc}")
        return {}


def _patch_config(**fields: Any) -> None:
    """Update selected top-level settings without replacing other settings."""
    ensure_config_dir()
    data = load_settings()
    data.update(fields)
    CONFIG_FILE.write_text(
        json.dumps(data, indent=4),
        encoding="utf-8",
    )


def _patch_flag(key: str, value: Any) -> None:
    _patch_config(**{key: value})


def config_exists() -> bool:
    return CONFIG_FILE.exists()


# ---------------------------------------------------------------------------
# Ollama-only local model configuration — no provider selection or registry.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

def get_ollama_config() -> dict:
    data = load_settings()
    return {
        "base_url": str(data.get("ollama_url") or data.get("llm_url") or DEFAULT_OLLAMA_URL).rstrip("/"),
        "model": str(data.get("ollama_model") or data.get("llm_model") or DEFAULT_OLLAMA_MODEL),
        "timeout": int(data.get("ollama_timeout", 180) or 180),
    }

def get_ollama_url() -> str:
    return get_ollama_config()["base_url"]

def get_ollama_model() -> str:
    return get_ollama_config()["model"]

def save_ollama_config(base_url: str = None, model: str = None, timeout: int = None) -> None:
    values = {}
    if base_url is not None:
        url = str(base_url).strip().rstrip("/")
        if not url.startswith("http://127.0.0.1") and not url.startswith("http://localhost"):
            raise ValueError("Ollama must run locally on this computer.")
        values["ollama_url"] = url
        values["llm_url"] = url
    if model is not None:
        model = str(model).strip()
        if not model: raise ValueError("Ollama model cannot be empty.")
        values["ollama_model"] = model
        values["llm_model"] = model
    if timeout is not None:
        values["ollama_timeout"] = max(1, min(600, int(timeout)))
    if values: _patch_config(**values)

def is_ollama_available() -> bool:
    import urllib.error
    import urllib.request
    try:
        request = urllib.request.Request(f"{get_ollama_url()}/api/tags", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False

def is_configured() -> bool:
    return is_ollama_available()


# Assistant identity
# ---------------------------------------------------------------------------

def get_assistant_name() -> str:
    return load_settings().get("assistant_name", "JARVIS") or "JARVIS"


def get_user_name() -> str:
    return load_settings().get("user_name", "MV") or "MV"


def save_assistant_config(
    assistant_name: str,
    user_name: str,
) -> None:
    _patch_config(
        assistant_name=(assistant_name or "").strip() or "JARVIS",
        user_name=(user_name or "").strip(),
    )


# ---------------------------------------------------------------------------
# Local TTS voice preference
# ---------------------------------------------------------------------------

DEFAULT_VOICE = "default"


def get_voice() -> str:
    """
    Return a local TTS voice preference.

    Voice availability depends on the selected TTS engine and operating system.
    """
    return str(load_settings().get("voice_name", DEFAULT_VOICE) or DEFAULT_VOICE)


def save_voice(voice_name: str) -> None:
    _patch_config(
        voice_name=str(voice_name or DEFAULT_VOICE).strip() or DEFAULT_VOICE
    )


# ---------------------------------------------------------------------------
# Wake word and push-to-talk
# ---------------------------------------------------------------------------

def get_wake_word_enabled() -> bool:
    return bool(load_settings().get("wake_word_enabled", True))


def save_wake_word_enabled(enabled: bool) -> None:
    _patch_flag("wake_word_enabled", bool(enabled))


def get_push_to_talk_enabled() -> bool:
    return bool(load_settings().get("push_to_talk_enabled", True))


def save_push_to_talk_enabled(enabled: bool) -> None:
    _patch_flag("push_to_talk_enabled", bool(enabled))


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

HUD_STYLES = ("face", "core")


def get_hud_style() -> str:
    style = str(load_settings().get("hud_style", "core")).strip().lower()
    return "core"


def save_hud_style(style: str) -> None:
    style = str(style or "").strip().lower()
    _patch_flag("hud_style", "core")


# ---------------------------------------------------------------------------
# Optional local assistant settings
# ---------------------------------------------------------------------------

def get_thinking_enabled() -> bool:
    """Return the optional local model reasoning preference."""
    return bool(load_settings().get("thinking_enabled", False))


def save_thinking_enabled(enabled: bool) -> None:
    _patch_flag("thinking_enabled", bool(enabled))


def get_turn_tuning() -> dict:
    """Local microphone turn-detection preferences."""
    cfg = load_settings().get("turn_tuning")
    cfg = cfg if isinstance(cfg, dict) else {}

    def bounded_int(key, default, minimum, maximum):
        try:
            return max(minimum, min(maximum, int(cfg.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "silence_ms": bounded_int("silence_ms", 550, 200, 3000),
        "prefix_ms": bounded_int("prefix_ms", 150, 0, 1000),
        "end_sensitivity": str(
            cfg.get("end_sensitivity", "high")
        ).lower(),
        "start_sensitivity": str(
            cfg.get("start_sensitivity", "default")
        ).lower(),
    }


def save_turn_tuning(values: dict) -> None:
    data = load_settings()
    current = data.get("turn_tuning")
    current = dict(current) if isinstance(current, dict) else {}
    current.update(values or {})
    _patch_config(turn_tuning=current)



MEDIA_RESOLUTIONS = ("default", "low", "medium", "high")


def get_media_resolution() -> str:
    value = str(
        load_settings().get("media_resolution", "medium")
    ).strip().lower()
    return value if value in MEDIA_RESOLUTIONS else "medium"


def save_media_resolution(value: str) -> None:
    value = str(value or "").strip().lower()
    _patch_flag(
        "media_resolution",
        value if value in MEDIA_RESOLUTIONS else "medium",
    )


def get_brief_enabled() -> bool:
    return bool(load_settings().get("morning_brief_enabled", False))


def save_brief_enabled(enabled: bool) -> None:
    _patch_flag("morning_brief_enabled", bool(enabled))


# ---------------------------------------------------------------------------
# Audio devices
# ---------------------------------------------------------------------------

def get_input_device() -> str:
    return str(load_settings().get("input_device", "") or "").strip()


def save_input_device(name: str) -> None:
    _patch_config(input_device=(name or "").strip())


def get_output_device() -> str:
    return str(load_settings().get("output_device", "") or "").strip()


def save_output_device(name: str) -> None:
    _patch_config(output_device=(name or "").strip())

