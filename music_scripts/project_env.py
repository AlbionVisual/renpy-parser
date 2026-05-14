import os
from pathlib import Path

from dotenv import load_dotenv

MUSIC_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = MUSIC_SCRIPTS_DIR.parent


def _quiet_essentia_warnings_if_requested() -> None:
    v = os.environ.get("ESSENTIA_SILENCE_WARNINGS", "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    import essentia

    essentia.log.warningActive = False


def ensure_dotenv_loaded() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
    _quiet_essentia_warnings_if_requested()
    return REPO_ROOT
