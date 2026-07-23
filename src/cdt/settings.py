"""Settings for the Commercial Debt Tracker project."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def resolve_path(path: Path) -> Path:
    """Resolve a path to an absolute path.

    If the path is not absolute, it is assumed to be relative to the project root.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# Load environment variables from .env file if it exists
load_dotenv()

# Set the data directory
DATA_DIR = resolve_path(Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_TOKEN"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", "openai/gpt-5.4")
EXTRACTOR_REASONING = os.environ.get("EXTRACTOR_REASONING", "none")
# OpenAI Batch API uses native model ids (no provider prefix) and a reasoning_effort
# vocabulary distinct from OpenRouter's. These configure the deployed batch backend.
EXTRACTOR_BATCH_MODEL = os.environ.get("EXTRACTOR_BATCH_MODEL", "gpt-5.4")
EXTRACTOR_BATCH_REASONING = os.environ.get("EXTRACTOR_BATCH_REASONING", "none")
