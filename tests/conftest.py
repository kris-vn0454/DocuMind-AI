"""Shared pytest fixtures."""
import pytest
from pathlib import Path
import tempfile


@pytest.fixture(scope="session")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
