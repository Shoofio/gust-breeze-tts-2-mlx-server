"""The runtime seam's shared rules and types (specs/005-mlx-mac-inference/contracts/runtime-seam.md).

What the routes, synthesis and voice code need besides the runtime itself: the chunk it yields,
its configuration, the CFG and prompt-length rules every room check uses, and the range checks
for per-request sampling overrides. `models/mlx_streaming.py` implements the seam on top of these.

Plain Python and numpy (no torch, no mlx), so the model-free tests import it cheaply. These rules
were first written for the upstream CUDA runtime, which the MLX runtime was built to match; the
names are kept from there (`FastStreamingConfig`, `select_fast_cfg`, ...).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np

# --- Sampling settings -------------------------------------------------------------------

# Temperatures. MAX_TEMPERATURE is the HTTP contract's limit (0, 10], which the runtime's
# overrides enforce. MIN_TEMPERATURE is a numeric floor, not a limit: below 1e-5, logits / t
# overflows float32 to inf and softmax turns it into NaN, while sampling there is already
# effectively greedy, so any smaller positive temperature is raised to it.
MIN_TEMPERATURE = 1e-5
MAX_TEMPERATURE = 10.0

# The repetition penalties the runtime accepts, the HTTP contract's [1e-4, 10]. A positive logit
# is divided by the penalty, so 1e-40 overflows it to inf (then NaN); at 1e-4 a logit of 30
# becomes 3e5, still finite.
MIN_REPETITION_PENALTY = 1e-4
MAX_REPETITION_PENALTY = 10.0


class NonFiniteLogitsError(RuntimeError):
    """Logits that cannot be sampled: NaN, +inf, or no finite value in a row.

    A model or numerics failure, never the client's input (every sampling setting is
    range-checked first), so it is a RuntimeError and not a ValueError that a route could map
    to 400.
    """


_NONFINITE_MESSAGE = "logits contain NaN or +inf, or no finite value; cannot sample"


class NumberRule(NamedTuple):
    """What a sampling setting accepts: an integer or a real, within a range."""

    integer: bool
    minimum: float
    minimum_inclusive: bool
    maximum: float = math.inf

    def describe(self) -> str:
        kind = "an integer" if self.integer else "a finite number"
        low = "[" if self.minimum_inclusive else "("
        high = "inf)" if self.maximum == math.inf else f"{self.maximum:g}]"
        return f"{kind} in {low}{self.minimum:g}, {high}"


def is_valid_number(value: Any, rule: NumberRule) -> bool:
    """Whether ``value`` satisfies ``rule``; never raises.

    Any ``numbers.Integral``/``numbers.Real`` counts (so numpy scalars do), but not a bool. NaN
    fails both range comparisons. The range is checked before ``math.isfinite``, which raises
    OverflowError for an int too large for a float (10**400) while the comparisons are exact.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, numbers.Integral if rule.integer else numbers.Real):
        return False
    above = value >= rule.minimum if rule.minimum_inclusive else value > rule.minimum
    if not (above and value <= rule.maximum):
        return False
    if rule.integer:
        return True
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def require_number(name: str, value: Any, rule: NumberRule, *, suffix: str = "") -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value`` satisfies ``rule``.

    The one number check for sampling settings. ``suffix`` extends the expectation (such as
    " or None").
    """
    if not is_valid_number(value, rule):
        raise ValueError(f"{name} must be {rule.describe()}{suffix}, got {value!r}")


REPETITION_PENALTY_RULE = NumberRule(
    False, MIN_REPETITION_PENALTY, True, MAX_REPETITION_PENALTY
)

# Per-request overrides. Counts must be integers (a fraction is a caller bug, not something to
# truncate). temperature and repetition_penalty take the HTTP contract's ranges; top_p needs no
# upper bound here, since above 1 it filters nothing.
_OVERRIDE_RULES = {
    "temperature": NumberRule(False, 0.0, False, MAX_TEMPERATURE),
    "top_k": NumberRule(True, 0, False),
    "top_p": NumberRule(False, 0.0, False),
    "repetition_penalty": REPETITION_PENALTY_RULE,
    "max_new_tokens": NumberRule(True, 0, False),
}
# A configured or model default top_k of 0 means "no top-k filtering".
_DEFAULT_TOP_K_RULE = NumberRule(True, 0, True)
_POSITIVE_INTEGER_RULE = NumberRule(True, 0, False)
# The guidance scale: 0 runs the negative prompt alone, 1 disables CFG. 100 is far past any
# useful scale (SillyTavern sends up to 10) and keeps the cond - uncond difference it multiplies
# from overflowing.
_CFG_SCALE_RULE = NumberRule(False, 0.0, True, 100.0)


def _require_valid_overrides(**overrides: Any) -> None:
    """Reject a per-request override that is neither ``None`` nor valid (``_OVERRIDE_RULES``).

    ``None`` means the default. Translating the wire's "use the default" values (such as
    ``max_new_tokens: 0``) into ``None`` is the API boundary's job, so anything else invalid
    here is a caller bug and is not coerced: a non-number or a bool raises ``ValueError`` like
    any other bad value.
    """
    for name, value in overrides.items():
        if value is not None:
            require_number(name, value, _OVERRIDE_RULES[name], suffix=" or None")


# --- Reference prefixes ------------------------------------------------------------------

# Frames of audio a registered voice prefix must leave room for: 12 frames at the codec's
# 12.5 Hz is about 1 s.
MIN_SUFFIX_FRAMES = 12
# Cache slots a reference prefix must leave free: the suffix of a one-word text with the default
# instruction on the guided branch (10 tokens, measured with prepare_suffix_inputs and the real
# tokenizer for "a", "Hi", "I", "嗨" and ".") plus MIN_SUFFIX_FRAMES, since the decode loop needs
# prefix_len + suffix_len + frames <= max_seq_len - 1.
MIN_SUFFIX_ROOM = 10 + MIN_SUFFIX_FRAMES


def max_reference_prefix_len(max_seq_len: int) -> int:
    """The longest reference prefix ``build_reference_prefix`` accepts in a
    ``max_seq_len``-token context: one that leaves ``MIN_SUFFIX_ROOM`` slots. The one statement
    of that rule, which the voice routes also check before a voice is registered and before a
    request takes the GPU."""
    return max_seq_len - 1 - MIN_SUFFIX_ROOM


# --- Configuration and output ------------------------------------------------------------


@dataclass(frozen=True)
class FastStreamingConfig:
    """The runtime's limits and sampling defaults (`runtime.config`).

    `max_new_tokens` is the hard ceiling on frames per request; the default when a request sets
    none is the model's generation_config.max_new_tokens, clamped to this.
    """

    max_new_tokens: int = 1500
    max_seq_len: int = 1024
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    repetition_penalty: float = 1.1

    def __post_init__(self) -> None:
        # Fail at construction, not mid-stream: every value here reaches the sampler or sizes a
        # buffer on every request.
        require_number("max_new_tokens", self.max_new_tokens, _POSITIVE_INTEGER_RULE)
        require_number("max_seq_len", self.max_seq_len, _POSITIVE_INTEGER_RULE)
        require_number(
            "repetition_penalty",
            self.repetition_penalty,
            _OVERRIDE_RULES["repetition_penalty"],
        )
        for name, value, rule in (
            ("temperature", self.temperature, _OVERRIDE_RULES["temperature"]),
            ("top_k", self.top_k, _DEFAULT_TOP_K_RULE),
            ("top_p", self.top_p, _OVERRIDE_RULES["top_p"]),
        ):
            if value is not None:
                require_number(name, value, rule)


@dataclass(frozen=True)
class FastStreamingChunk:
    """One streamed piece of audio: float32 samples for `codec_frames` codec frames."""

    audio: np.ndarray
    sample_rate: int
    codec_frames: int
    is_final: bool
    timing: dict[str, float | int | bool] = field(default_factory=dict)


class NoRoomError(ValueError):
    """A prompt (plus any reference prefix and CFG) already fills the context, so no frame at
    all can be generated: the same condition as ``max_new_tokens_room(...) <= 0``, raised
    instead of returned when it is only discovered once ``iter_audio_chunks`` starts.

    Its own class because a caller maps it to a `400`, while every other ``ValueError`` the
    runtime raises is a caller or programming error and must stay a `500`.
    """


# --- CFG and prompt length ---------------------------------------------------------------

FastCfgMode = Literal["no_cfg", "single_cfg"]

_DUAL_CFG_KEYS = (
    "cfg_scale_ref",
    "cfg_scale_ins",
    "cfg_uncond_prompt_ids",
    "cfg_ref_prompt_ids",
    "cfg_ins_prompt_ids",
)


@dataclass(frozen=True)
class FastCfgSelection:
    mode: FastCfgMode
    guidance_scale: float
    use_negative_as_main: bool


def reject_dual_cfg(inputs: dict[str, Any]) -> None:
    present = [key for key in _DUAL_CFG_KEYS if inputs.get(key) is not None]
    if present:
        raise ValueError(
            "fast streaming supports only no_cfg and single_cfg; "
            f"dual CFG fields are not supported: {present}"
        )


def select_fast_cfg(inputs: dict[str, Any]) -> FastCfgSelection:
    """Which rows run for ``inputs``: no CFG, single CFG, or (``cfg_scale`` 0) the negative
    prompt alone. Validates ``cfg_scale`` where it is first read."""
    reject_dual_cfg(inputs)
    cfg_scale = inputs.get("cfg_scale", 1.0)
    require_number("cfg_scale", cfg_scale, _CFG_SCALE_RULE)
    cfg_scale = float(cfg_scale)
    has_negative = inputs.get("cfg_negative_prompt_ids") is not None
    if cfg_scale == 0.0 and has_negative:
        return FastCfgSelection(
            mode="no_cfg", guidance_scale=1.0, use_negative_as_main=True
        )
    if cfg_scale != 1.0:
        if not has_negative:
            raise ValueError(
                "single_cfg requires cfg_negative_prompt_ids when cfg_scale != 1.0"
            )
        return FastCfgSelection(
            mode="single_cfg", guidance_scale=cfg_scale, use_negative_as_main=False
        )
    return FastCfgSelection(
        mode="no_cfg", guidance_scale=1.0, use_negative_as_main=False
    )


class _BranchShape(NamedTuple):
    cfg: FastCfgSelection
    branch_batch_size: int
    seq_len: int


def _branch_shape(inputs: dict[str, Any]) -> _BranchShape:
    """``(cfg, branch_batch_size, seq_len)`` of the backbone batch for ``inputs``.

    The one place that says which prompts run: ``cfg_scale`` 0 runs the negative prompt alone,
    CFG runs both rows, padded to the longer one.
    """
    cfg = select_fast_cfg(inputs)
    if cfg.use_negative_as_main:
        return _BranchShape(
            cfg, 1, int(inputs["cfg_negative_prompt_attention_mask"].shape[1])
        )
    if cfg.mode == "no_cfg":
        return _BranchShape(cfg, 1, int(inputs["attention_mask"].shape[1]))
    seq_len = max(
        int(inputs["attention_mask"].shape[1]),
        int(inputs["cfg_negative_prompt_attention_mask"].shape[1]),
    )
    return _BranchShape(cfg, 2, seq_len)


class PromptLength(NamedTuple):
    """What a prompt's room depends on: the backbone batch it runs as and its length."""

    branch_batch_size: int
    seq_len: int


def prompt_length(inputs: dict[str, Any]) -> PromptLength:
    """``inputs``' `PromptLength`, for sizing a prompt by length alone
    (``room_for_length``) without keeping its tensors."""
    _, branch_batch_size, seq_len = _branch_shape(inputs)
    return PromptLength(branch_batch_size, seq_len)
