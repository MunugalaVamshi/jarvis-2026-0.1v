
"""
youtube_video.py — JARVIS YouTube Tool
AI summarization: Local Ollama only.
No Gemini, OpenAI, Groq, or other hosted AI model APIs.

Requirements:
    pip install requests youtube-transcript-api

Install Ollama separately, then run:
    ollama pull llama3.2

Default Ollama endpoint:
    http://localhost:11434
"""

import json
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus, urlparse

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    requests = None
    _REQUESTS_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _TRANSCRIPT_OK = True
except ImportError:
    YouTubeTranscriptApi = None
    _TRANSCRIPT_OK = False


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 120

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_YT_VIDEO_FILTER = "EgIQAQ%3D%3D"


# ---------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------

def _get_ollama_settings() -> tuple[str, str, int]:
    """
    Read Ollama settings from config/settings.json.

    Expected configuration:
    providers.creds.ollama.base_url
    providers.creds.ollama.model
    providers.creds.ollama.timeout
    """
    base_url = OLLAMA_DEFAULT_URL
    model = OLLAMA_DEFAULT_MODEL
    timeout = OLLAMA_TIMEOUT

    try:
        if SETTINGS_PATH.exists():
            with SETTINGS_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)

            providers = config.get("providers", {})
            creds = providers.get("creds", {})
            ollama = creds.get("ollama", {})

            if isinstance(ollama, dict):
                base_url = ollama.get("base_url") or base_url
                model = ollama.get("model") or model

                try:
                    timeout = int(ollama.get("timeout", timeout))
                except (TypeError, ValueError):
                    timeout = OLLAMA_TIMEOUT

    except (OSError, json.JSONDecodeError) as exc:
        print(f"[YouTube] Could not read Ollama settings: {exc}")

    return base_url.rstrip("/"), model, timeout


def _summarize_with_ollama(transcript: str, video_url: str) -> str:
    """Summarize a YouTube transcript using the local Ollama API only."""
    if not _REQUESTS_OK:
        raise RuntimeError(
            "The requests package is missing. Install it with: pip install requests"
        )

    base_url, model, timeout = _get_ollama_settings()

    # Limit input size to reduce memory and context-limit problems.
    max_chars = 24000
    truncated = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated += "\n[Transcript shortened because it was too long.]"

    system_prompt = (
        "You are JARVIS, a helpful AI assistant. "
        "Summarize YouTube transcripts accurately. "
        "Start with a one-sentence overview, followed by 3-5 key points. "
        "Mention important steps, facts, examples, and conclusions. "
        "Do not invent information absent from the transcript. "
        "Address the user as 'sir' when natural. "
        "Respond in the same language as the transcript."
    )

    user_prompt = (
        f"Summarize the following YouTube video transcript.\n"
        f"Video URL: {video_url}\n\n"
        f"TRANSCRIPT:\n{truncated}"
    )

    endpoint = f"{base_url}/api/chat"

    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 8192,
                },
            },
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        summary = (
            data.get("message", {}).get("content", "")
            if isinstance(data, dict)
            else ""
        )

        if not summary.strip():
            raise RuntimeError("Ollama returned an empty summary.")

        return summary.strip()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to local Ollama. Make sure Ollama is running "
            "and available at " + base_url
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama took longer than {timeout} seconds to respond."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        details = ""
        try:
            details = response.text[:500]
        except Exception:
            pass

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' may not be installed. "
                f"Run: ollama pull {model}"
            ) from exc

        raise RuntimeError(
            f"Ollama HTTP error {response.status_code}: {details}"
        ) from exc


# ---------------------------------------------------------
# Browser and YouTube helpers
# ---------------------------------------------------------

def _open_url(url: str) -> None:
    """Open a URL in the default browser."""
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"[YouTube] Could not open URL: {exc}")


def _scrape_first_video_url(query: str) -> str | None:
    if not _REQUESTS_OK:
        return None

    search_url = (
        "https://www.youtube.com/results"
        f"?search_query={quote_plus(query)}"
        f"&sp={_YT_VIDEO_FILTER}"
    )

    try:
        response = requests.get(
            search_url, headers=HEADERS, timeout=15
        )
        response.raise_for_status()
        html = response.text

        video_ids = re.findall(
            r'"videoId":"([A-Za-z0-9_-]{11})"', html
        )

        seen = set()

        for video_id in video_ids:
            if video_id in seen:
                continue
            seen.add(video_id)

            # Skip IDs that appear in a Shorts URL.
            if re.search(
                rf"/shorts/{re.escape(video_id)}(?:[?\"/]|$)", html
            ):
                continue

            return f"https://www.youtube.com/watch?v={video_id}"

    except Exception as exc:
        print(f"[YouTube] Search failed: {exc}")

    return None


def _extract_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()

        if host in ("youtu.be", "www.youtu.be"):
            candidate = parsed.path.strip("/").split("/")[0]

        elif host in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
        ):
            if parsed.path == "/watch":
                from urllib.parse import parse_qs
                candidate = parse_qs(parsed.query).get("v", [""])[0]
            else:
                match = re.match(
                    r"^/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})",
                    parsed.path,
                )
                candidate = match.group(1) if match else ""
        else:
            return None

        return (
            candidate
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate)
            else None
        )

    except Exception:
        return None


def _is_valid_youtube_url(url: str) -> bool:
    return _extract_video_id(url or "") is not None


def _ask_for_url(
    prompt_text: str = "YouTube video URL:"
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk._default_root
        created_root = root is None

        if created_root:
            root = tk.Tk()
            root.withdraw()

        try:
            url = simpledialog.askstring(
                "J.A.R.V.I.S", prompt_text, parent=root
            )
            return url.strip() if url else None
        finally:
            if created_root:
                root.destroy()

    except Exception as exc:
        print(f"[YouTube] URL dialog failed: {exc}")
        return None


# ---------------------------------------------------------
# Transcript retrieval
# ---------------------------------------------------------

def _get_transcript(video_id: str) -> str | None:
    if not _TRANSCRIPT_OK:
        return None

    try:
        # Support newer and older youtube-transcript-api interfaces.
        api = YouTubeTranscriptApi()

        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        else:
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                video_id
            )

        transcript = None

        languages = [
            "en", "te", "hi", "tr", "de", "fr", "es",
            "it", "pt", "ru", "ja", "ko", "ar", "zh",
        ]

        try:
            transcript = transcript_list.find_manually_created_transcript(
                languages
            )
        except Exception:
            pass

        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(
                    languages
                )
            except Exception:
                pass

        if transcript is None:
            try:
                transcript = next(iter(transcript_list), None)
            except Exception:
                transcript = None

        if transcript is None:
            return None

        fetched = transcript.fetch()
        pieces = []

        for entry in fetched:
            if isinstance(entry, dict):
                text = entry.get("text", "")
            else:
                text = getattr(entry, "text", "")

            if text:
                pieces.append(text)

        return " ".join(pieces).strip() or None

    except Exception as exc:
        print(f"[YouTube] Transcript retrieval failed: {exc}")
        return None


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

def _save_summary(content: str, video_url: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"youtube_summary_{ts}.txt"

    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.home()

    filepath = desktop / filename

    header = (
        "JARVIS — YouTube Summary\n"
        f"{'─' * 50}\n"
        f"URL    : {video_url}\n"
        f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{'─' * 50}\n\n"
    )

    filepath.write_text(header + content, encoding="utf-8")

    try:
        subprocess.Popen(["notepad.exe", str(filepath)])
    except Exception as exc:
        print(f"[YouTube] Could not open Notepad: {exc}")

    return str(filepath)


# ---------------------------------------------------------
# Video information
# ---------------------------------------------------------

def _scrape_video_info(video_id: str) -> dict:
    if not _REQUESTS_OK:
        return {}

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        html = response.text

        info = {}

        patterns = [
            ("title", r'"title":\{"runs":\[\{"text":"([^"]+)"'),
            ("channel", r'"ownerChannelName":"([^"]+)"'),
            ("views", r'"viewCount":"(\d+)"'),
            ("duration", r'"lengthSeconds":"(\d+)"'),
            ("likes", r'"label":"([0-9,]+ likes)"'),
        ]

        for key, pattern in patterns:
            match = re.search(pattern, html)

            if not match:
                continue

            raw = match.group(1)

            if key == "views":
                info[key] = f"{int(raw):,}"
            elif key == "duration":
                seconds = int(raw)
                hours, remainder = divmod(seconds, 3600)
                minutes, secs = divmod(remainder, 60)

                info[key] = (
                    f"{hours}:{minutes:02d}:{secs:02d}"
                    if hours
                    else f"{minutes}:{secs:02d}"
                )
            else:
                info[key] = raw

        return info

    except Exception as exc:
        print(f"[YouTube] Video info retrieval failed: {exc}")
        return {}


# ---------------------------------------------------------
# Trending videos
# ---------------------------------------------------------

def _scrape_trending(
    region: str = "IN",
    max_results: int = 8,
) -> list[dict]:
    if not _REQUESTS_OK:
        return []

    region = re.sub(r"[^A-Z]", "", region.upper())[:2] or "IN"
    url = f"https://www.youtube.com/feed/trending?gl={region}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        html = response.text

        titles = re.findall(
            r'"title":\{"runs":\[\{"text":"([^"]+)"\}\]', html
        )
        channels = re.findall(
            r'"ownerText":\{"runs":\[\{"text":"([^"]+)"', html
        )

        results = []
        seen = set()

        for index, title in enumerate(titles):
            if title in seen or len(title.strip()) < 5:
                continue

            seen.add(title)
            channel = (
                channels[index] if index < len(channels) else "Unknown"
            )

            results.append({
                "rank": len(results) + 1,
                "title": title,
                "channel": channel,
            })

            if len(results) >= max_results:
                break

        return results

    except Exception as exc:
        print(f"[YouTube] Trending retrieval failed: {exc}")
        return []


# ---------------------------------------------------------
# Actions
# ---------------------------------------------------------

def _handle_play(parameters: dict, player) -> str:
    query = str(parameters.get("query", "")).strip()

    if not query:
        return "Please tell me what you'd like to watch, sir."

    if player:
        player.write_log(f"[YouTube] Searching: {query}")

    video_url = _scrape_first_video_url(query)

    if video_url:
        _open_url(video_url)
        return f"Playing: {query}"

    fallback_url = (
        "https://www.youtube.com/results"
        f"?search_query={quote_plus(query)}"
        f"&sp={_YT_VIDEO_FILTER}"
    )
    _open_url(fallback_url)

    return f"Opened YouTube search for: {query}"


def _handle_summarize(
    parameters: dict,
    player,
    speak,
) -> str:
    if not _TRANSCRIPT_OK:
        return (
            "youtube-transcript-api is not installed. "
            "Run: pip install youtube-transcript-api"
        )

    url = str(parameters.get("url", "")).strip()

    if not url:
        url = _ask_for_url("Please paste the YouTube video URL:")

    if not url:
        return "No URL provided, sir. Summary cancelled."

    video_id = _extract_video_id(url)

    if not video_id:
        return "That doesn't appear to be a valid YouTube URL, sir."

    if player:
        player.write_log(f"[YouTube] Summarizing: {url}")

    if speak:
        speak("Fetching the transcript now, sir. One moment.")

    transcript = _get_transcript(video_id)

    if not transcript:
        return (
            "I couldn't retrieve a transcript. "
            "The video may not have captions available."
        )

    if speak:
        speak("Transcript retrieved. Summarizing with local Ollama.")

    try:
        summary = _summarize_with_ollama(transcript, url)
    except Exception as exc:
        return f"Local Ollama summarization failed, sir: {exc}"

    if speak:
        speak(summary)

    if parameters.get("save", False):
        saved_path = _save_summary(summary, url)
        return f"Summary saved to: {saved_path}\n\n{summary}"

    return summary


def _handle_get_info(
    parameters: dict,
    player,
    speak,
) -> str:
    url = str(parameters.get("url", "")).strip()

    if not url:
        url = _ask_for_url("Please paste the YouTube video URL:")

    if not url:
        return "Please provide a YouTube URL, sir."

    video_id = _extract_video_id(url)

    if not video_id:
        return "That doesn't appear to be a valid YouTube URL, sir."

    if player:
        player.write_log(f"[YouTube] Getting info: {url}")

    info = _scrape_video_info(video_id)

    if not info:
        return "Could not retrieve video information, sir."

    lines = [
        f"{key.capitalize()}: {info[key]}"
        for key in ("title", "channel", "views", "duration", "likes")
        if key in info
    ]

    result = "\n".join(lines) or "No video information was found."

    if speak:
        speak("Here's the video information, sir. " + result.replace("\n", ". "))

    return result


def _handle_trending(
    parameters: dict,
    player,
    speak,
) -> str:
    region = str(parameters.get("region", "IN")).upper()

    if player:
        player.write_log(f"[YouTube] Trending: {region}")

    trending = _scrape_trending(region=region, max_results=8)

    if not trending:
        return (
            f"Could not retrieve trending videos for {region}, sir. "
            "YouTube's page structure or regional availability may have changed."
        )

    lines = [f"Trending videos in {region}:"]
    lines.extend(
        f"{video['rank']}. {video['title']} — {video['channel']}"
        for video in trending
    )

    result = "\n".join(lines)

    if speak:
        top3 = trending[:3]
        spoken = "Here are the trending videos, sir. " + ". ".join(
            f"Number {video['rank']}: {video['title']} by {video['channel']}"
            for video in top3
        )
        speak(spoken)

    return result


_ACTION_MAP = {
    "play": _handle_play,
    "summarize": _handle_summarize,
    "get_info": _handle_get_info,
    "trending": _handle_trending,
}


def youtube_video(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None,
) -> str:
    params = parameters or {}
    action = str(params.get("action", "play")).lower().strip()

    if player:
        player.write_log(f"[YouTube] Action: {action}")

    print(f"[YouTube] Action: {action}; Params: {params}")

    handler = _ACTION_MAP.get(action)

    if handler is None:
        return (
            f"Unknown YouTube action: '{action}'. "
            "Available actions: play, summarize, get_info, trending."
        )

    try:
        if action == "play":
            return handler(params, player) or "Done."

        return handler(params, player, speak) or "Done."

    except Exception as exc:
        print(f"[YouTube] Error in {action}: {exc}")
        return f"YouTube {action} failed, sir: {exc}"


# ---------------------------------------------------------
# Tool declaration — discovered by core/action_loader.py
# ---------------------------------------------------------

TOOL = {
    "name": "youtube_video",
    "description": (
        "Controls YouTube: plays videos, summarizes video transcripts "
        "using local Ollama, retrieves video information, and lists "
        "trending videos. Actions: play, summarize, get_info, trending."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": (
                    "play | summarize | get_info | trending "
                    "(default: play)"
                ),
            },
            "query": {
                "type": "STRING",
                "description": "Search query for the play action.",
            },
            "save": {
                "type": "BOOLEAN",
                "description": "Save the summary to a text file.",
            },
            "region": {
                "type": "STRING",
                "description": "Country code for trending videos, e.g. IN or US.",
            },
            "url": {
                "type": "STRING",
                "description": (
                    "YouTube URL for summarize or get_info. "
                    "If omitted, a URL dialog opens."
                ),
            },
        },
        "required": [],
    },
    "handler": youtube_video,
}


# ---------------------------------------------------------
# Tool declaration — discovered by core/action_loader.py
# ---------------------------------------------------------

TOOL = {
    "name": "youtube_video",
    "description": (
        "Controls YouTube: plays videos, summarizes video transcripts "
        "using local Ollama, retrieves video information, and lists "
        "trending videos. Actions: play, summarize, get_info, trending."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": (
                    "play | summarize | get_info | trending"
                ),
            },
            "query": {
                "type": "STRING",
                "description": "Search query for the play action.",
            },
            "save": {
                "type": "BOOLEAN",
                "description": "Save the summary to a text file.",
            },
            "region": {
                "type": "STRING",
                "description": "Country code for trending videos.",
            },
            "url": {
                "type": "STRING",
                "description": "YouTube URL for summarize or get_info.",
            },
        },
        "required": [],
    },
    "handler": youtube_video,
}
