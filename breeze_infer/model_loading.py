"""Load and warm up the streaming runtime from `Settings` (one job: model loading).

Everything here runs on the `GpuThread` (research.md R14), in the background, while `/health`
answers `503 loading`. Nothing prints: the composition root turns the returned report into the
`model.loaded` event.
"""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from breeze_infer.settings import Settings


@dataclass(frozen=True)
class LoadedModel:
    """The ready runtime plus the facts worth reporting in `model.loaded`.

    `cpu_tokenizer` and `sizing_tokenizer` are two copies of the runtime's tokenizer, for the
    speech route's two CPU workers (`routes_speech.CpuTokenizer`): the room check before the
    gate, and the lease holder's anchor sizing. Neither may share the GPU thread's tokenizer,
    nor each other, since all three threads can tokenize at once. Required: the speech route
    cannot size a request without them (`api.Components.mark_ready` installs them together
    with the runtime). `from_runtime` makes the copies.
    """

    runtime: Any
    report: dict[str, Any]
    cpu_tokenizer: Any
    sizing_tokenizer: Any

    @classmethod
    def from_runtime(cls, runtime: Any, report: dict[str, Any]) -> LoadedModel:
        """`runtime` with its own tokenizer copied twice, once for each CPU worker.

        Call it on the GPU thread before the server reports ready, while nothing else uses the
        tokenizer: a deep copy of a real one takes about 700 ms, which would otherwise stall
        the event loop (`CpuTokenizer.install`, from `mark_ready`) or, made lazily, the first
        speech request.
        """
        return cls(
            runtime=runtime,
            report=report,
            cpu_tokenizer=copy.deepcopy(runtime.tokenizer),
            sizing_tokenizer=copy.deepcopy(runtime.tokenizer),
        )


def load_model(
    settings: Settings, device: str, environ: MutableMapping[str, str]
) -> LoadedModel:
    """Load the MLX checkpoint and compile its Metal kernels before the server reports ready.

    `environ` is the process environment, passed in by the composition root; the MLX speed
    settings (`BREEZE_MLX_*`, `MlxSpeedOptions.from_env`) are read from it. `models.mlx_streaming`
    is imported here, not at the top: it needs mlx, which only a Mac has, and the model-free
    tests import this module everywhere.
    """
    from models.mlx_streaming import MlxSpeedOptions, load_mlx_runtime

    runtime = load_mlx_runtime(settings.model_path, MlxSpeedOptions.from_env(environ))
    warmup_ms = runtime.warmup()
    report: dict[str, Any] = {
        "backend": "mlx",
        "weights": settings.weights,
        "device": device,
        "warmup_ms": round(warmup_ms, 2),
        "speed": runtime.speed.report(),
    }
    # On the GPU thread, before the server reports ready (`LoadedModel.from_runtime`).
    return LoadedModel.from_runtime(runtime, report)
