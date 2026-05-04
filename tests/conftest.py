"""Shared fixtures for all tests."""

from pathlib import Path

import pytest

# Resolve sample data directory relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = _REPO_ROOT / "data" / "sample" / "T35_RH20"


@pytest.fixture
def pomiar1_path() -> Path:
    return SAMPLE_DIR / "Pomiar1.txt"


@pytest.fixture
def pomiar3_path() -> Path:
    return SAMPLE_DIR / "Pomiar3.txt"


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE_DIR
