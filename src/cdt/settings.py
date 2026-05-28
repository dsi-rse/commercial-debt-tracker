"""Settings for the Commercial Debt Tracker project."""

import os
from pathlib import Path

from dotenv import load_dotenv


def resolve_path(path: Path) -> Path:
    """Resolve a path to an absolute path.

    If the path is not absolute, it is assumed to be relative to the project root.
    """
    path = path.expanduser()
    if not path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        path = project_root / path
    return path.resolve()


# Load environment variables from .env file if it exists
load_dotenv()

# Set the data directory
DATA_DIR = resolve_path(Path(os.environ["DATA_DIR"]))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_TOKEN"
)
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", "openai/gpt-5.4")
EXTRACTOR_REASONING = os.environ.get("EXTRACTOR_REASONING", "none")
