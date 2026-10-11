import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data(*paths): needs the real datasets at these repo-relative paths; skipped when absent",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        marker = item.get_closest_marker("requires_data")
        if marker is None:
            continue
        missing = [p for p in marker.args if not (REPO_ROOT / p).exists()]
        if missing:
            item.add_marker(pytest.mark.skip(reason=f"real data not present: {', '.join(missing)}"))
