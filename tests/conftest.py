"""Run every test against a scratch copy of the repo's data folders, never the real ones."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="ercot-test-"))
os.environ["ERCOT_ROOT"] = str(_TMP)
os.environ.pop("EIA_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src import config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_dirs():
    for d in (config.FORECASTS, config.OUTCOMES, config.NOTES, config.DATA,
              config.BACKTEST, config.CACHE):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    for f in (config.SCORECARD, config.DOCS / "data.json"):
        if f.exists():
            f.unlink()
    from src import sources
    sources.FEED.clear()
    yield
