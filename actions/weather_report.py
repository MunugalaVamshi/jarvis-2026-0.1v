
"""
weather_action.py — JARVIS Weather Search using Local Ollama

AI model: Ollama running locally
Live information: Google Search opened in the browser
No Gemini, OpenAI, or external weather API.
"""

import json
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 60


def _get_ollama_config() -> dict:
    settings = {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            settings = json.load(file)
    except (OSError, json.JSONDecodeError):
        pass

    providers = settings.get("providers", {})
    credentials = providers.get("creds", {})
    ollama = credentials.get("ollama", {})

    return {
        "base_url": ollama.get(
            "base_url", DEFAULT_OLLAMA_URL
        ).rstrip("/"),
        "model": ollama.get(
            "model", DEFAULT_OLLAMA_MODEL
        ),
        "timeout": int(
            ollama.get("timeout", DEFAULT_TIMEOUT)
        ),
    }


# ---------------------------------------------------------------------------
# Local Ollama
# ---------------------------------------------------------------------------

def _interpret_weather_request(city: str, when: str) -> str:
    """
    Use local Ollama to interpret the requested city and time.
    Does not fetch weather measurements.
    """
    config = _get_ollama_config()

    prompt = (
        "You are JARVIS, a concise weather-search assistant. "
        "Create a short Google search query for the requested weather. "
        "Return only the search query, without quotation marks or explanation.\n"
        f"City: {city}\n"
        f"Requested time: {when}\n"
        "Include the city and requested time."
    )

    response = requests.post(
        f'{config["base_url"]}/api/chat',
        json={
            "model": config["model"],
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "options": {
                "temperature": 0,
            },
        },
        timeout=config["timeout"],
    )

    response.raise_for_status()

    answer = response.json().get("message", {}).get(
        "content", ""
    ).strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer[:300]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(message: str, player=None) -> None:
    print(f"[Weather] {message}")

    if player:
        try:
            player.write_log(f"JARVIS: {message}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public action
# ---------------------------------------------------------------------------

def weather_action(
    parameters: dict,
    player=None,
    session_memory=None,
) -> str:
    parameters = parameters or {}

    city = parameters.get("city")
    when = parameters.get("time", "today")

    if not isinstance(city, str) or not city.strip():
        message = "Sir, please provide a city for the weather report."
        _log(message, player)
        return message

    if not isinstance(when, str) or not when.strip():
        when = "today"

    city = city.strip()
    when = when.strip()

    try:
        # Ollama runs locally and prepares the search query.
        search_query = _interpret_weather_request(city, when)

    except requests.exceptions.ConnectionError:
        # Still allow the weather search if Ollama is not running.
        _log("Ollama is unavailable; using the original search query.", player)
        search_query = f"weather in {city} {when}"

    except requests.exceptions.Timeout:
        _log("Ollama timed out; using the original search query.", player)
        search_query = f"weather in {city} {when}"

    except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
        _log(f"Ollama request failed: {exc}", player)
        search_query = f"weather in {city} {when}"

    # Google Search retrieves the live information in the browser.
    url = (
        "https://www.google.com/search?q="
        + quote_plus(search_query)
    )

    try:
        opened = webbrowser.open(url)

        if not opened:
            raise RuntimeError("The browser could not be opened.")

    except Exception as exc:
        message = f"Sir, I couldn't open the weather search: {exc}"
        _log(message, player)
        return message

    message = f"Showing the weather search for {city}, {when}, sir."
    _log(message, player)

    if session_memory:
        try:
            session_memory.set_last_search(
                query=search_query,
                response=message,
            )
        except Exception:
            pass

    return message


# ---------------------------------------------------------------------------
# JARVIS tool registration
# ---------------------------------------------------------------------------

TOOL = {
    "name": "weather_report",
    "description": (
        "Search for weather information for a city and requested time. "
        "Uses local Ollama to prepare the query and opens Google Search "
        "to view current weather information."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "city": {
                "type": "STRING",
                "description": "City name, for example Hyderabad.",
            },
            "time": {
                "type": "STRING",
                "description": "Requested time, such as today or tomorrow.",
            },
        },
        "required": ["city"],
    },
    "handler": weather_action,
}
