import sys
from pathlib import Path

# `shared/` (preprocessing used by both api/ and model/) sits at the repo root locally and at
# /app/shared in the Docker image, where PYTHONPATH already covers it.
try:
    import shared  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
