"""Load project .env regardless of current working directory."""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent


def load_app_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
