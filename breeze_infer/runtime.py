"""The sampling defaults Breeze TTS 2 is served with, from upstream's inference code."""

from __future__ import annotations

from typing import Any


def update_generation_config_for_breeze(
    model: Any,
    generation_config: dict[str, Any] | None = None,
) -> None:
    """Apply the Breeze sampling defaults to ``model.generation_config`` and
    ``model.depth_decoder.generation_config``, as upstream's inference code does.

    The built-in values override the checkpoint's ``generation_config.json`` for every sampling
    setting of both, and for the backbone's ``max_new_tokens``.
    """
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)
