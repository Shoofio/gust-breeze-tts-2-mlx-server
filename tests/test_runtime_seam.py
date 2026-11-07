"""The runtime seam's shared rules (models/runtime_seam.py): configuration checks, CFG selection,
prompt length and the per-request override ranges. No model, no mlx."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.runtime_seam import (
    MAX_REPETITION_PENALTY,
    MAX_TEMPERATURE,
    MIN_REPETITION_PENALTY,
    MIN_SUFFIX_FRAMES,
    MIN_SUFFIX_ROOM,
    FastStreamingConfig,
    NonFiniteLogitsError,
    NoRoomError,
    NumberRule,
    PromptLength,
    _branch_shape,
    _require_valid_overrides,
    max_reference_prefix_len,
    prompt_length,
    reject_dual_cfg,
    require_number,
    select_fast_cfg,
)


def _prompt(length: int, negative_length: int | None = None) -> dict:
    inputs = {"attention_mask": torch.ones(1, length, dtype=torch.long)}
    if negative_length is not None:
        inputs.update(
            cfg_scale=2.0,
            cfg_negative_prompt_ids=torch.ones(1, negative_length, dtype=torch.long),
            cfg_negative_prompt_attention_mask=torch.ones(
                1, negative_length, dtype=torch.long
            ),
        )
    return inputs


# --- FastStreamingConfig ---


def test_config_defaults_to_repetition_penalty_1p1() -> None:
    assert FastStreamingConfig().repetition_penalty == 1.1


def test_config_ceiling_defaults_to_the_server_maximum() -> None:
    # R12: 750 is the per-request default (from the model); 1500 the ceiling.
    assert FastStreamingConfig().max_new_tokens == 1500


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": float("nan")},
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"top_p": float("nan")},
        {"top_k": -1},
        {"top_k": 2.5},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("nan")},
        {"repetition_penalty": 1e-40},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_seq_len": 0},
    ],
)
def test_config_rejects_invalid_fields(fields) -> None:
    with pytest.raises(ValueError, match=next(iter(fields))):
        FastStreamingConfig(**fields)


def test_config_accepts_top_k_zero_as_disabled() -> None:
    assert FastStreamingConfig(top_k=0).top_k == 0


# --- CFG selection ---


def test_cfg_selects_no_cfg_by_default() -> None:
    cfg = select_fast_cfg({})

    assert cfg.mode == "no_cfg"
    assert cfg.guidance_scale == 1.0
    assert cfg.use_negative_as_main is False


def test_cfg_selects_single_cfg_when_negative_prompt_is_present() -> None:
    cfg = select_fast_cfg(
        {"cfg_scale": 2.5, "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    assert cfg.mode == "single_cfg"
    assert cfg.guidance_scale == 2.5
    assert cfg.use_negative_as_main is False


def test_cfg_zero_uses_negative_as_main() -> None:
    cfg = select_fast_cfg(
        {"cfg_scale": 0.0, "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    assert cfg.mode == "no_cfg"
    assert cfg.guidance_scale == 1.0
    assert cfg.use_negative_as_main is True


def test_cfg_other_than_one_needs_a_negative_prompt() -> None:
    with pytest.raises(ValueError, match="cfg_negative_prompt_ids"):
        select_fast_cfg({"cfg_scale": 2.0})


@pytest.mark.parametrize("cfg_scale", [0.0, 0, 2.5, 100.0, np.float32(7.5)])
def test_cfg_accepts_scales_from_0_to_100(cfg_scale) -> None:
    cfg = select_fast_cfg(
        {"cfg_scale": cfg_scale, "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    if cfg_scale != 0:
        assert cfg.guidance_scale == pytest.approx(float(cfg_scale))


@pytest.mark.parametrize(
    "cfg_scale",
    [float("nan"), float("inf"), -0.5, 100.5, 10**400, "4", True, None],
)
def test_cfg_rejects_scales_outside_0_to_100(cfg_scale) -> None:
    with pytest.raises(ValueError, match="cfg_scale"):
        select_fast_cfg(
            {"cfg_scale": cfg_scale, "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long)}
        )


def test_dual_cfg_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="dual CFG"):
        reject_dual_cfg({"cfg_scale_ref": 1.0, "cfg_scale_ins": 2.0})


# --- Prompt length ---


@pytest.mark.parametrize(
    ("inputs", "shape"),
    [
        (_prompt(7), ("no_cfg", 1, 7)),
        # CFG pads both branches to the longer one.
        (_prompt(7, 9), ("single_cfg", 2, 9)),
        (_prompt(9, 7), ("single_cfg", 2, 9)),
        # cfg_scale 0 runs the negative prompt alone.
        ({**_prompt(7, 9), "cfg_scale": 0.0}, ("no_cfg", 1, 9)),
    ],
)
def test_branch_shape(inputs, shape) -> None:
    cfg, branch_batch_size, seq_len = _branch_shape(inputs)

    assert (cfg.mode, branch_batch_size, seq_len) == shape


def test_prompt_length_is_the_branch_batch_and_the_longer_row() -> None:
    assert prompt_length(_prompt(7)) == PromptLength(branch_batch_size=1, seq_len=7)
    assert prompt_length(_prompt(7, 9)) == PromptLength(branch_batch_size=2, seq_len=9)


# --- Reference prefixes ---


def test_min_suffix_room_leaves_about_a_second_of_audio() -> None:
    # 12 frames at the codec's 12.5 Hz is about 1 s.
    assert MIN_SUFFIX_FRAMES == 12
    assert MIN_SUFFIX_ROOM > MIN_SUFFIX_FRAMES


def test_longest_reference_prefix_leaves_the_min_suffix_room() -> None:
    assert max_reference_prefix_len(2048) == 2048 - 1 - MIN_SUFFIX_ROOM


# --- Errors and number rules ---


def test_runtime_limits_match_the_http_contract() -> None:
    # contracts/http-api.md: temperature (0, 10], repetition_penalty [1e-4, 10].
    assert MAX_TEMPERATURE == 10
    assert MAX_REPETITION_PENALTY == 10
    assert MIN_REPETITION_PENALTY == 1e-4


def test_nonfinite_logits_error_is_a_server_error_not_a_bad_request() -> None:
    # A route maps ValueError to 400; a model producing NaN is not the client's fault.
    assert issubclass(NonFiniteLogitsError, RuntimeError)
    assert not issubclass(NonFiniteLogitsError, ValueError)


def test_no_room_is_a_value_error_of_its_own() -> None:
    assert issubclass(NoRoomError, ValueError)
    assert NoRoomError is not ValueError


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        (1.0, True),
        (np.float32(0.5), True),
        (0.0, False),
        (float("nan"), False),
        (10**400, False),
        (True, False),
        ("1", False),
    ],
)
def test_require_number_is_the_one_shared_check(value, valid) -> None:
    rule = NumberRule(integer=False, minimum=0.0, minimum_inclusive=False)
    if valid:
        require_number("x", value, rule)
    else:
        with pytest.raises(ValueError, match="x must be"):
            require_number("x", value, rule)


_INVALID_OVERRIDES = [
    {"temperature": 0.0},
    {"top_k": 0},
    {"top_p": -0.1},
    {"repetition_penalty": 0.0},
    {"max_new_tokens": 0},
    {"max_new_tokens": -3},
    {"temperature": float("nan")},
    {"temperature": float("inf")},
    {"top_p": float("nan")},
    {"top_p": float("inf")},
    {"repetition_penalty": float("nan")},
    {"repetition_penalty": float("inf")},
    {"top_k": float("nan")},
    {"max_new_tokens": float("inf")},
    # Counts must be integers; the runtime does not truncate them.
    {"top_k": 2.5},
    {"max_new_tokens": 2.5},
    # Absurd magnitudes are a caller error, not a numerics problem.
    {"temperature": 1e39},
    {"temperature": MAX_TEMPERATURE * (1 + 1e-9)},
    {"repetition_penalty": MAX_REPETITION_PENALTY * (1 + 1e-9)},
    {"temperature": 1e4},
    {"repetition_penalty": 1e4},
    # Non-numbers and bools are rejected as ValueError, never TypeError.
    {"temperature": "0.5"},
    {"top_p": "x"},
    {"repetition_penalty": [1.1]},
    {"top_k": "5"},
    {"max_new_tokens": "5"},
    {"temperature": True},
    {"top_p": True},
    {"repetition_penalty": False},
    {"top_k": True},
    {"max_new_tokens": True},
    {"temperature": np.bool_(True)},
    {"top_k": np.bool_(True)},
    # Below the floor a penalty overflows the logits it divides.
    {"repetition_penalty": 1e-40},
    {"repetition_penalty": MIN_REPETITION_PENALTY * (1 - 1e-9)},
    # Too large for a float: rejected, not an OverflowError.
    {"temperature": 10**400},
    {"top_p": 10**400},
    {"repetition_penalty": -(10**400)},
]


@pytest.mark.parametrize("override", _INVALID_OVERRIDES)
def test_invalid_overrides_are_rejected(override) -> None:
    with pytest.raises(ValueError, match=next(iter(override))):
        _require_valid_overrides(**override)


def test_valid_overrides_pass_including_numpy_scalars_and_none() -> None:
    _require_valid_overrides(
        temperature=np.float32(0.5),
        top_k=np.int64(7),
        top_p=np.float64(0.9),
        repetition_penalty=np.float32(1.25),
        max_new_tokens=np.int64(3),
    )
    _require_valid_overrides(
        temperature=None, top_k=None, top_p=None, repetition_penalty=None, max_new_tokens=None
    )
    _require_valid_overrides(temperature=MAX_TEMPERATURE, repetition_penalty=MIN_REPETITION_PENALTY)
