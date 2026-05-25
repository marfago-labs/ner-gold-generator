"""Load project ``.env`` before CLI or library use."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def load_project_env(*, env_path: Path | None = None) -> bool:
    """Load environment variables from a ``.env`` file once per process.

    Uses ``python-dotenv`` when installed. Subsequent calls are no-ops.

    Args:
        env_path: Explicit ``.env`` path. Defaults to the repository root
            ``.env`` next to ``pyproject.toml``.

    Returns:
        ``True`` if a file existed and was loaded; ``False`` if the file was
        missing, dotenv is unavailable, or env was already loaded.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return False
    path = env_path or (_PROJECT_ROOT / ".env")
    if not path.is_file():
        _ENV_LOADED = True
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        _ENV_LOADED = True
        return False
    load_dotenv(path, override=False)
    _ENV_LOADED = True
    return True
