import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The app resolves presets/static/templates relative to the repo root
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep the API tests independent from a developer's local auth settings
os.environ.pop("AUTH_USERNAME", None)
os.environ.pop("AUTH_PASSWORD", None)


@pytest.fixture(scope="session")
def decision_engine():
    from app.decision import DecisionEngine
    return DecisionEngine(presets_dir="app/presets", alert_threshold=0.80)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
