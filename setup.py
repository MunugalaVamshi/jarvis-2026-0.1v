
"""
MV — one-time setup for local Ollama JARVIS.

Installs Python dependencies for the current operating system,
then installs the Playwright browsers used for web automation.

LLM:
  - Ollama running locally at http://localhost:11434
  - Default model: llama3.2
  - No hosted LLM API keys required.

Speech:
  - STT: local Whisper, if configured in requirements.txt
  - TTS: local Piper or supported operating-system voices

Optional:
  - Wake word: enable from ⚙ → WAKE WORD inside the app.
  - Avatar: uses the shipped face model and installed graphics libraries.

NOTE:
This script installs dependencies and checks setup requirements.
It does not automatically rewrite main.py or provider configuration.
"""

import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OS = platform.system()
HERE = Path(__file__).resolve().parent

MIN_PY = (3, 11)
MAX_PY = (3, 13)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def _run(label: str, args: list[str]) -> None:
    print(f"\n▶ {label}")
    subprocess.run(args, check=True)


def _check_python() -> None:
    """Check the supported Python version."""
    version = sys.version_info[:2]

    if version > MAX_PY:
        print(
            f"\n⚠️ Python {version[0]}.{version[1]} is newer than "
            f"the tested version {MAX_PY[0]}.{MAX_PY[1]}. Continuing."
        )
    elif version < MIN_PY:
        print(
            f"\n❌ Python {version[0]}.{version[1]} detected. "
            f"MV requires Python {MIN_PY[0]}.{MIN_PY[1]} or newer."
        )
        print("Install a supported Python version and rerun setup.")
        sys.exit(1)


def _check_assets() -> None:
    """Warn if the optional avatar model is missing."""
    face = HERE / "core" / "face_model.obj"

    if not face.exists() or face.stat().st_size < 4096:
        print(
            "\n⚠️ core/face_model.obj is missing or truncated.\n"
            "   The avatar may fall back to the plain glowing core."
        )


def _check_ollama() -> bool:
    """Check whether the local Ollama service responds."""
    try:
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/tags",
            headers={"Accept": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200

    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _check_ollama_model() -> bool:
    """Check whether the default local model is installed."""
    import json

    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")

        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        models = data.get("models", [])
        available = {
            model.get("name", "")
            for model in models
        }

        return any(
            name == DEFAULT_MODEL or name.startswith(DEFAULT_MODEL + ":")
            for name in available
        )

    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _check_local_ai() -> None:
    """Report Ollama status without requesting any hosted API keys."""
    print("\n── Local AI check ──")

    if not _check_ollama():
        print("⚠️ Ollama is not responding at", OLLAMA_URL)
        print("   Install Ollama and start the Ollama application/service.")
        print("   Download: https://ollama.com/download")
        print(
            f"   Then run: ollama pull {DEFAULT_MODEL}"
        )
        return

    print("✅ Ollama is running locally.")

    if _check_ollama_model():
        print(f"✅ Local model '{DEFAULT_MODEL}' is installed.")
    else:
        print(f"⚠️ Local model '{DEFAULT_MODEL}' was not found.")
        print(f"   Run: ollama pull {DEFAULT_MODEL}")


def main() -> None:
    print(
        f"⚙ MV local setup — OS: {OS or 'unknown'}, "
        f"Python {sys.version_info[0]}.{sys.version_info[1]}"
    )

    _check_python()

    requirements = HERE / "requirements.txt"

    if not requirements.exists():
        print("\n❌ requirements.txt was not found.")
        sys.exit(1)

    _run(
        "Installing Python dependencies...",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
        ],
    )

    try:
        _run(
            "Installing Playwright browsers...",
            [
                sys.executable,
                "-m",
                "playwright",
                "install",
                "chromium",
                "firefox",
            ],
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"\n⚠️ Playwright browser installation failed: {error}")
        print(
            "   Retry later with: "
            "python -m playwright install chromium firefox"
        )

    _check_assets()

    if OS == "Windows":
        try:
            import win32com.client  # noqa: F401
        except ImportError:
            print(
                "\n⚠️ pywin32 is unavailable. Some Windows integrations "
                "may use fallback behavior."
            )
            print(
                f'   Run: "{sys.executable}" -m pip install pywin32'
            )

    elif OS == "Linux":
        print(
            "\nℹ️ Linux optional system tools:\n"
            "   • Volume: pulseaudio-utils (pactl)\n"
            "   • Brightness: brightnessctl\n"
            "   • Open URLs: xdg-utils\n"
            "   • Reminders: systemd or at"
        )

    _check_local_ai()

    print("\n✅ Setup finished.")
    print("   1) Ensure Ollama is installed and running.")
    print(f"   2) Ensure your local model is installed: {DEFAULT_MODEL}")
    print("   3) Launch JARVIS: python main.py")
    print("   4) Wake word defaults to ON; first launch may prepare its local model.")
    print("\n   No hosted AI API key is required. Ollama runs locally.")


if __name__ == "__main__":
    main()
