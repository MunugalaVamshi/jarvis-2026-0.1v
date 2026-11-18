"""
core/platform_support.py — detect supported platform capabilities.

Supported operating systems:
    Windows
    Linux

Unsupported operating systems:
    macOS and all others
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field


@dataclass
class Support:
    """What is possible on this machine."""

    os_name: str
    can_control_input: bool = True
    can_capture_screen: bool = True
    can_focus_windows: bool = True
    notes: list[str] = field(default_factory=list)
    supported: bool = True

    @property
    def fully_capable(self) -> bool:
        return (
            self.supported
            and self.can_control_input
            and self.can_capture_screen
            and self.can_focus_windows
        )


def _linux_session() -> str:
    """Return 'wayland', 'x11', or '' if no graphical session is found."""
    session = (os.environ.get("XDG_SESSION_TYPE") or "").strip().lower()

    if session in ("wayland", "x11"):
        return session

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"

    if os.environ.get("DISPLAY"):
        return "x11"

    return ""


def detect() -> Support:
    """Detect supported operating systems and their capabilities."""

    system = platform.system()

    # Windows support
    if system == "Windows":
        return Support(os_name="Windows")

    # Reject macOS and every OS other than Windows or Linux
    if system != "Linux":
        return Support(
            os_name=system or "Unknown",
            can_control_input=False,
            can_capture_screen=False,
            can_focus_windows=False,
            notes=[
                f"{system or 'Unknown'} is not supported. "
                "This application supports Windows and Linux only."
            ],
            supported=False,
        )

    # Linux support
    support = Support(os_name="Linux")
    session = _linux_session()

    if not session:
        support.can_control_input = False
        support.can_capture_screen = False
        support.can_focus_windows = False
        support.notes.append(
            "No graphical session was found. Keyboard control, mouse control, "
            "screen capture and window focusing are unavailable."
        )
        return support

    if session == "wayland":
        support.can_control_input = False
        support.can_capture_screen = False
        support.can_focus_windows = False
        support.notes.append(
            "Wayland restrictions may prevent keyboard and mouse control, "
            "screen capture and window focusing. Some actions may require "
            "desktop-specific permissions or alternative implementations."
        )
        return support

    # Linux X11 session
    if not (shutil.which("xdotool") or shutil.which("wmctrl")):
        support.can_focus_windows = False
        support.notes.append(
            "Neither xdotool nor wmctrl is installed. Focusing an existing "
            "application window may be unavailable."
        )

    return support


def describe_for_model(support: Support) -> list[str]:
    """Describe platform restrictions to the assistant model."""

    lines: list[str] = []

    if not support.supported:
        lines.append(
            f"- This operating system ({support.os_name}) is unsupported. "
            "This application supports Windows and Linux only. "
            "Do not claim that platform-dependent actions were performed."
        )
        return lines

    if not support.can_control_input:
        lines.append(
            "- Keyboard, mouse and input control are unavailable in this "
            "session. Do not claim to have performed these actions."
        )

    if not support.can_capture_screen:
        lines.append(
            "- Screen capture is unavailable in this session. "
            "Do not claim to have seen the screen."
        )

    if not support.can_focus_windows and support.can_control_input:
        lines.append(
            "- Focusing an already-running application may be unavailable. "
            "Do not claim that its window was brought to the front."
        )

    return lines


def describe_for_user(support: Support) -> list[str]:
    """Return startup notes for the activity log."""

    return [f"SYS: {note}" for note in support.notes]