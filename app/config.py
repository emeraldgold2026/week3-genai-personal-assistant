"""Loads API keys from a local .env file. Never logs or exposes key values."""
import os

from dotenv import load_dotenv

load_dotenv()


class MissingAPIKeyError(Exception):
    """Raised when a required API key is not set in the environment."""


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY not set — add it to your .env file and restart."
        )
    return key


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY not set — add it to your .env file and restart."
        )
    return key
