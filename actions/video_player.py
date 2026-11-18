
"""Play a video on the JARVIS HUD.

No Gemini/OpenAI calls. Ollama can select this tool through the main
assistant, but video playback itself does not require an AI model.
"""

from __future__ import annotations

import re
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

_YT = re.compile(
    r"(?:youtube\.com/|youtu\.be/)",
    re.IGNORECASE,
)

_open_lock = threading.Lock()
_open_seq = 0


def _begin_open() -> int:
    global _open_seq
    with _open_lock:
        _open_seq += 1
        return _open_seq


def _still_wanted(token: int) -> bool:
    with _open_lock:
        return token == _open_seq


def _is_url(text: str) -> bool:
    return text.lower().startswith(
        ("http://", "https://", "file://")
    )


def _local_path(text: str) -> str:
    raw = (text or "").strip().strip('"').strip("'")
    if not raw:
        return ""

    try:
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            return str(path)
    except (OSError, ValueError):
        pass

    return ""


def _resolve_youtube(source: str) -> tuple[str, str, str, str]:
    """Return video URL, audio URL, title and error."""

    try:
        import yt_dlp
    except ImportError:
        return (
            "", "", "",
            "yt-dlp is not installed. Run: pip install yt-dlp",
        )

    query = (
        source
        if _YT.search(source)
        else f"ytsearch1:{source}"
    )

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": "best",
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(query, download=False)

        if info and info.get("entries"):
            info = next(
                (entry for entry in info["entries"] if entry),
                None,
            )

        if not info:
            return "", "", "", "No video was found."

        formats = [
            item for item in info.get("formats", [])
            if item.get("url")
            and str(item.get("protocol", "")).startswith("http")
        ]

        def has_codec(item: dict, key: str) -> bool:
            return item.get(key) not in (None, "none")

        def best(items: list[dict]) -> dict | None:
            if not items:
                return None
            return max(
                items,
                key=lambda item: (
                    item.get("language_preference") or 0,
                    item.get("height") or 0,
                    item.get("tbr") or 0,
                ),
            )

        # Prefer a combined stream when available.
        combined = best([
            item for item in formats
            if has_codec(item, "vcodec")
            and has_codec(item, "acodec")
        ])

        if combined:
            return (
                combined["url"],
                "",
                info.get("title", "YouTube video"),
                "",
            )

        # Otherwise choose separate video and audio streams.
        video = best([
            item for item in formats
            if has_codec(item, "vcodec")
            and not has_codec(item, "acodec")
            and item.get("ext") == "mp4"
            and (item.get("height") or 0) <= 720
        ])

        if not video:
            video = best([
                item for item in formats
                if has_codec(item, "vcodec")
                and not has_codec(item, "acodec")
            ])

        audio = best([
            item for item in formats
            if has_codec(item, "acodec")
            and not has_codec(item, "vcodec")
            and item.get("ext") == "m4a"
        ])

        if not audio:
            audio = best([
                item for item in formats
                if has_codec(item, "acodec")
                and not has_codec(item, "vcodec")
            ])

        if not video:
            return "", "", "", "No playable video stream was found."

        return (
            video["url"],
            audio["url"] if audio else "",
            info.get("title", "YouTube video"),
            "",
        )

    except Exception as exc:
        return "", "", "", f"Could not resolve the video: {exc}"


def _search_page(query: str) -> str:
    return (
        "https://www.youtube.com/results?search_query="
        + quote_plus(query)
    )


def video_player(
    parameters: dict | None = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    action = str(params.get("action") or "play").strip().lower()
    source = str(params.get("source") or "").strip()

    if player is None:
        return "There is no display available for video playback."

    if action in ("stop", "close", "hide"):
        was_playing = player.video_is_playing()
        _begin_open()  # Invalidate any pending YouTube request.
        player.stop_video()
        return "Closed the video." if was_playing else "Cancelled pending playback."

    if action in ("mute", "sound_off", "unmute", "sound_on"):
        if not player.video_is_playing():
            return "Nothing is playing."

        muted = action in ("mute", "sound_off")
        player.set_video_muted(muted)
        return "Muted the video." if muted else "Turned the sound on."

    if action != "play":
        return f"Unsupported video action: {action}"

    if not source:
        return "What video would you like me to play?"

    local = _local_path(source)
    if local:
        player.show_video(local, Path(local).name, muted=True)
        return f"Playing {Path(local).name} on the HUD, muted."

    # Direct media URLs go straight to the HUD.
    if _is_url(source) and not _YT.search(source):
        player.show_video(
            source,
            source.rsplit("/", 1)[-1][:44] or "Video",
            muted=True,
        )
        return "Opening the video on the HUD, muted."

    token = _begin_open()

    threading.Thread(
        target=_play_youtube,
        args=(player, source, token),
        daemon=True,
        name="video-open",
    ).start()

    return f"status=opening source={source}"


def _play_youtube(player, source: str, token: int) -> None:
    try:
        video_url, audio_url, title, error = _resolve_youtube(source)
    except Exception as exc:
        video_url = audio_url = title = ""
        error = str(exc)

    if not _still_wanted(token):
        return

    if video_url:
        try:
            player.show_video(
                video_url,
                title or source,
                muted=True,
                audio_source=audio_url or None,
            )
            return
        except Exception as exc:
            error = str(exc)

    # Fallback: open the YouTube page in the browser.
    page = source if _YT.search(source) else _search_page(source)

    try:
        webbrowser.open(page)
        fallback = " I opened it in your browser instead."
    except Exception:
        fallback = " I could not open the browser either."

    _say(
        player,
        f"Tell the user briefly that {source} could not be played "
        f"on the HUD. Reason: {error}.{fallback}",
    )


def _say(player, instruction: str) -> None:
    if hasattr(player, "request_say"):
        try:
            player.request_say(instruction)
        except Exception:
            pass


TOOL = {
    "name": "video_player",
    "description": (
        "Play a local video, direct video URL, YouTube link, or YouTube "
        "search on the JARVIS HUD. Supports stop, mute and unmute. "
        "Videos start muted. Use this tool for video playback, not "
        "for summarizing YouTube transcripts."
    ),
    "brief": (
        "Plays videos on the HUD and supports stop, mute and unmute."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["play", "stop", "mute", "unmute"],
                "description": "Playback action; defaults to play.",
            },
            "source": {
                "type": "STRING",
                "description": (
                    "Local file path, direct media URL, YouTube URL, "
                    "or search phrase. Required for play."
                ),
            },
        },
        "required": ["action"],
    },
    "handler": video_player,
}
