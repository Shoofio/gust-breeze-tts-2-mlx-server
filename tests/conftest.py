from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

MLX_ENV = "BREEZE_MLX_MODEL"
# The official PyTorch checkpoint (BreezeBlue/Breeze-TTS-2), optional: only its tokenizer files
# are read, to compare with the MLX checkpoint's.
OFFICIAL_ENV = "BREEZE_OFFICIAL_MODEL"


def _mlx_skip_reason() -> str | None:
    # MLX only runs on Apple Silicon, so the env var alone is not enough.
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "mlx tests need macOS on Apple Silicon"
    if not os.environ.get(MLX_ENV):
        return f"set {MLX_ENV}=/path/to/mlx/checkpoint to run mlx tests"
    return None


def pytest_collection_modifyitems(config, items) -> None:
    mlx_reason = _mlx_skip_reason()
    if mlx_reason:
        mlx_skip = pytest.mark.skip(reason=mlx_reason)
        for item in items:
            if "mlx" in item.keywords:
                item.add_marker(mlx_skip)


@pytest.fixture(scope="session")
def mlx_model() -> Path:
    value = os.environ.get(MLX_ENV)
    if not value:
        pytest.skip(f"{MLX_ENV} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{MLX_ENV}={value} is not a directory")
    return path


@pytest.fixture(scope="session")
def official_model() -> Path | None:
    # Tests that compare against the official checkpoint (tokenizer parity) skip themselves when
    # this is None.
    value = os.environ.get(OFFICIAL_ENV)
    return Path(value) if value else None
