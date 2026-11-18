
"""
core/memory_manager.py
JARVIS local memory manager — Ollama-compatible, no external AI APIs.

Storage: memory/long_term.json
LLM: Ollama is called by the separate LLM provider.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Optional, Any


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_PATH = MEMORY_DIR / "long_term.json"

_lock = RLock()

MAX_VALUE_LENGTH = 380
MEMORY_MAX_CHARS = 200_000
PROMPT_CORE_CHARS = 900
PROMPT_INDEX_CHARS = 420
PROMPT_MAX_PER_CATEGORY = 6
_SESSION_MAX = 3

VALID_CATEGORIES = {
    "identity",
    "preferences",
    "projects",
    "relationships",
    "wishes",
    "notes",
}

CATEGORY_LABELS = {
    "preferences": "Preferences",
    "projects": "Active projects / goals",
    "relationships": "People in their life",
    "wishes": "Wishes / plans",
    "notes": "Notes",
}

IDENTITY_FIELDS = [
    "name", "age", "birthday", "city",
    "job", "language", "school", "nationality",
]


def _empty_memory() -> dict:
    return {
        "identity": {},
        "preferences": {},
        "projects": {},
        "relationships": {},
        "wishes": {},
        "notes": {},
        "sessions": [],
    }


def load_memory() -> dict:
    """Load memory from JSON, preserving compatibility with older files."""
    with _lock:
        if not MEMORY_PATH.exists():
            return _empty_memory()

        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return _empty_memory()

            memory = _empty_memory()

            for category in VALID_CATEGORIES:
                value = data.get(category, {})
                memory[category] = value if isinstance(value, dict) else {}

            sessions = data.get("sessions", [])
            memory["sessions"] = sessions if isinstance(sessions, list) else []

            # Preserve unknown fields from an older memory file.
            for key, value in data.items():
                if key not in memory:
                    memory[key] = value

            return memory

        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Memory] Load error: {exc}")
            return _empty_memory()


def _all_entries(memory: dict) -> list[tuple]:
    entries = []

    for category, items in memory.items():
        if not isinstance(items, dict):
            continue

        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((category, key, entry))

    return entries


_trim_notifier = None


def set_trim_notifier(fn) -> None:
    """Register a callable to notify the UI when memory is trimmed."""
    global _trim_notifier
    _trim_notifier = fn


def _notify_trim(message: str) -> None:
    if _trim_notifier:
        try:
            _trim_notifier(message)
        except Exception as exc:
            print(f"[Memory] Notification error: {exc}")


def _trim_to_limit(memory: dict) -> dict:
    """Trim oldest entries only when the JSON store exceeds its size limit."""
    def size() -> int:
        return len(json.dumps(memory, ensure_ascii=False))

    if size() <= MEMORY_MAX_CHARS:
        return memory

    entries = _all_entries(memory)
    entries.sort(
        key=lambda item: (
            item[2].get("updated", "0000-00-00"),
            item[0],
            item[1],
        )
    )

    dropped = []

    for category, key, _entry in entries:
        if size() <= MEMORY_MAX_CHARS:
            break

        memory[category].pop(key, None)
        dropped.append(f"{category}/{key}")

    if dropped:
        print(f"[Memory] Removed {len(dropped)} old entries.")
        _notify_trim(
            f"SYS: Memory limit reached; removed {len(dropped)} old entries: "
            + ", ".join(dropped[:3])
        )

    return memory


def save_memory(memory: dict) -> None:
    """Atomically save the memory store to disk."""
    if not isinstance(memory, dict):
        return

    memory = _trim_to_limit(memory)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    content = json.dumps(memory, indent=2, ensure_ascii=False)
    temporary = MEMORY_PATH.with_suffix(".json.tmp")

    with _lock:
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(MEMORY_PATH)
        except OSError as exc:
            print(f"[Memory] Save error: {exc}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def _truncate_value(value: str) -> str:
    if len(value) > MAX_VALUE_LENGTH:
        return value[:MAX_VALUE_LENGTH].rstrip() + "…"
    return value


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False

    for key, value in updates.items():
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True

            if _recursive_update(target[key], value):
                changed = True

            continue

        raw = value.get("value") if isinstance(value, dict) else value

        if raw is None:
            continue

        new_value = _truncate_value(str(raw).strip())
        if not new_value:
            continue

        existing = target.get(key, {})
        old_value = (
            existing.get("value")
            if isinstance(existing, dict)
            else existing
        )

        if old_value != new_value:
            target[key] = {
                "value": new_value,
                "updated": datetime.now().strftime("%Y-%m-%d"),
            }
            changed = True

    return changed


def update_memory(memory_update: dict) -> dict:
    """Update multiple categories using nested dictionaries."""
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()

    memory = load_memory()

    if _recursive_update(memory, memory_update):
        save_memory(memory)
        print(f"[Memory] Saved: {list(memory_update.keys())}")

    return memory


def _entry_value(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("value", "") or "").strip()
    return str(entry or "").strip()


def _pretty(key: str) -> str:
    return str(key).replace("_", " ").strip()


def format_memory_for_prompt(memory: Optional[dict]) -> str:
    """
    Create a compact context block for the Ollama system prompt.
    This function formats memory; it does not make an AI API request.
    """
    if not memory:
        return ""

    lines = []
    identity = memory.get("identity", {}) or {}

    # Identity information
    for field in IDENTITY_FIELDS:
        value = _entry_value(identity.get(field))
        if not value:
            continue

        if field == "language":
            lines.append(
                f"Previously observed language: {value}. "
                "Always answer in the language of the user's current message."
            )
        else:
            lines.append(f"{field.title()}: {value}")

    for key, entry in identity.items():
        if key in IDENTITY_FIELDS:
            continue

        value = _entry_value(entry)
        if value:
            lines.append(f"{_pretty(key).title()}: {value}")

    # Most recently updated memories first
    recent = []

    for category in CATEGORY_LABELS:
        items = memory.get(category, {}) or {}

        if not isinstance(items, dict):
            continue

        for key, entry in items.items():
            value = _entry_value(entry)
            if not value:
                continue

            updated = (
                entry.get("updated", "")
                if isinstance(entry, dict)
                else ""
            ) or "0000-00-00"

            recent.append((updated, category, key, value))

    recent.sort(key=lambda row: row[0], reverse=True)

    used = sum(len(line) + 1 for line in lines)
    shown = {}
    overflow = {}
    counts = {}

    for _updated, category, key, value in recent:
        line = f"  - {_pretty(key).title()}: {value}"

        if (
            counts.get(category, 0) < PROMPT_MAX_PER_CATEGORY
            and used + len(line) + 1 <= PROMPT_CORE_CHARS
        ):
            shown.setdefault(category, []).append(line)
            counts[category] = counts.get(category, 0) + 1
            used += len(line) + 1
        else:
            overflow.setdefault(category, []).append(_pretty(key))

    # Interleave the index so categories get a fair share.
    indexed = []
    categories = [
        category for category in CATEGORY_LABELS
        if overflow.get(category)
    ]
    cursors = {category: 0 for category in categories}

    while categories:
        for category in list(categories):
            index = cursors[category]

            if index >= len(overflow[category]):
                categories.remove(category)
                continue

            indexed.append(overflow[category][index])
            cursors[category] += 1

    for category, label in CATEGORY_LABELS.items():
        if shown.get(category):
            lines.extend(["", f"{label}:"])
            lines.extend(shown[category])

    if not lines and not indexed:
        return ""

    output = [
        "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite as a list]",
        *lines,
    ]

    if indexed:
        names = []
        budget = PROMPT_INDEX_CHARS

        for name in indexed:
            cost = len(name) + 2
            if budget - cost < 0:
                break

            names.append(name)
            budget -= cost

        if names:
            output.extend([
                "",
                "[ALSO REMEMBERED — use recall_memory with a keyword to retrieve values]",
                ", ".join(names)
                + (
                    f" (+{len(indexed) - len(names)} more)"
                    if len(indexed) > len(names)
                    else ""
                ),
            ])

    return "\n".join(output) + "\n"


def _score(
    query_words: list[str],
    category: str,
    key: str,
    value: str,
) -> int:
    """Fast keyword scoring without embeddings or network calls."""
    key_text = _pretty(key).lower()
    value_text = value.lower()
    score = 0

    for word in query_words:
        if word == key_text:
            score += 10
        elif word in key_text:
            score += 6

        if word in value_text:
            score += 3

        if word in category:
            score += 1

    return score


def search_memory(query: str, limit: int = 8) -> str:
    """Search stored memories. Used by the recall_memory tool."""
    memory = load_memory()
    words = [
        word
        for word in re.split(r"[^\w]+", (query or "").lower())
        if len(word) > 1
    ]

    rows = []

    for category, items in memory.items():
        if not isinstance(items, dict):
            continue

        for key, entry in items.items():
            value = _entry_value(entry)
            if not value:
                continue

            score = _score(words, category, key, value) if words else 1

            if score > 0:
                rows.append((score, category, key, value))

    if not rows:
        return (
            f"Nothing stored about '{query}'."
            if query
            else "I have not stored anything about this person yet."
        )

    rows.sort(key=lambda row: (-row[0], row[2]))
    results = rows[:max(1, limit)]

    output = [
        f"{category}/{_pretty(key)}: {value}"
        for _score_value, category, key, value in results
    ]

    header = (
        f"Stored facts matching '{query}':"
        if query
        else "Everything currently stored:"
    )

    extra = (
        f"\n(+{len(rows) - len(output)} more — search with a narrower keyword)"
        if len(rows) > len(output)
        else ""
    )

    return header + "\n" + "\n".join(output) + extra


def all_entries_for_ui() -> list[dict]:
    """Return memory entries for the JARVIS UI panel."""
    memory = load_memory()
    rows = []

    for category, items in memory.items():
        if not isinstance(items, dict):
            continue

        for key, entry in items.items():
            value = _entry_value(entry)
            if not value:
                continue

            rows.append({
                "category": category,
                "key": key,
                "value": value,
                "updated": (
                    entry.get("updated", "")
                    if isinstance(entry, dict)
                    else ""
                ),
            })

    rows.sort(
        key=lambda row: row["updated"] or "0000-00-00",
        reverse=True,
    )

    return rows


def remember(key: str, value: str, category: str = "notes") -> str:
    """Save one memory."""
    if category not in VALID_CATEGORIES:
        category = "notes"

    key = (key or "").strip()
    value = (value or "").strip()

    if not key or not value:
        return "Memory not saved: key and value are required."

    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    """Delete one memory."""
    if category not in VALID_CATEGORIES:
        return f"Invalid category: {category}"

    memory = load_memory()
    items = memory.get(category, {})

    if key not in items:
        return f"Not found: {category}/{key}"

    del items[key]
    save_memory(memory)
    return f"Forgotten: {category}/{key}"


forget_memory = forget


def save_session_summary(summary: str, language: str = "") -> None:
    """Save a short summary of a JARVIS session."""
    summary = (summary or "").strip()

    if not summary:
        return

    memory = load_memory()
    sessions = memory.get("sessions", [])

    if not isinstance(sessions, list):
        sessions = []

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "summary": summary[:280],
    }

    if language:
        entry["language"] = language

    sessions.append(entry)
    memory["sessions"] = sessions[-_SESSION_MAX:]

    save_memory(memory)

    print(
        f"[Memory] Session saved ({entry['date']}): "
        f"{summary[:60]}…"
    )


def pop_last_session() -> Optional[dict]:
    """Return and remove the latest session summary."""
    with _lock:
        if not MEMORY_PATH.exists():
            return None

        try:
            memory = json.loads(
                MEMORY_PATH.read_text(encoding="utf-8")
            )

            if not isinstance(memory, dict):
                return None

            sessions = memory.get("sessions", [])

            if not isinstance(sessions, list) or not sessions:
                return None

            entry = sessions.pop()
            memory["sessions"] = sessions

            save_memory(memory)
            return entry

        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Memory] pop_last_session error: {exc}")
            return None


# ---------------------------------------------------------------------------
# Optional Ollama helper
# ---------------------------------------------------------------------------

def ask_ollama_with_memory(
    user_message: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """
    Optional standalone Ollama chat helper.

    Requires:
        pip install requests
        ollama pull llama3.2

    If your project already has an Ollama provider, use that provider instead
    of duplicating this helper.
    """
    import requests

    memory_context = format_memory_for_prompt(load_memory())

    system_prompt = (
        "You are JARVIS, a helpful local AI assistant. "
        "Use relevant saved memories naturally. "
        "Do not invent facts about the user. "
        "When asked to remember something, use the application's memory "
        "tools if available; this chat helper does not automatically save it."
    )

    if memory_context:
        system_prompt += "\n\n" + memory_context

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=timeout,
    )

    response.raise_for_status()
    payload = response.json()

    return payload["message"]["content"]
