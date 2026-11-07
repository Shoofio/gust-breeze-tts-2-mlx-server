"""Launch configuration: a frozen ``Settings`` value object built from argv.

Per data-model.md "Settings", this is built once at the composition root from
the command line. It never reads ``os.environ`` (Principle III: environment
resolution belongs to the composition root, not to argument parsing).
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from breeze_infer.origins import canonical_origin

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SPLIT_CHARS = 600
DEFAULT_CHUNK_FIRST = 1
DEFAULT_CHUNK_MAX = 25
DEFAULT_VOICES_DIR = Path("voices")

# The smallest Apple Silicon Mac with room for the bf16 weights, the codec and the KV cache
# (research R5).
MLX_MIN_MEMORY_BYTES = 16 * 1024**3

# Named in the refusal of a checkpoint that isn't MLX. The launcher holds the repository ids and
# revisions, in one place, and prints the download command when a snapshot is missing.
MLX_WEIGHTS_HINT = "scripts/start_breeze_mac.sh names the download command for each precision"

# The refusal of --backend cuda, which earlier versions of this server had.
CUDA_REMOVED = (
    "--backend cuda was removed: this server runs on Apple Silicon (MLX) only. For NVIDIA GPUs, "
    "see northcraftfoundries/breeze-tts2-fast-streaming-api or breezeblue-ai/breeze-tts"
)

_PYTORCH_MODEL_TYPE = "breeze"
_MLX_MODEL_TYPE = "breeze_tts"

# Sentinel distinguishing "--ws-port" absent (derive port + 1) from an
# explicit "--ws-port disabled" (which resolves to None on Settings).
_WS_PORT_UNSET = object()


@dataclass(frozen=True)
class Platform:
    """The machine facts the startup checks depend on, read once at the entry point.

    Passed into ``settings_from_args`` rather than read there, so the parser stays a pure
    function of its arguments and tests build a ``Platform`` directly instead of patching.
    """

    system: str
    machine: str
    memory_bytes: int

    @classmethod
    def detect(cls) -> Platform:
        # The memory check only applies on macOS; elsewhere the Apple Silicon check refuses the
        # launch first, so memory is left at 0.
        memory_bytes = 0
        if sys.platform == "darwin":
            memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return cls(system=sys.platform, machine=platform_module.machine(), memory_bytes=memory_bytes)

    @property
    def is_macos(self) -> bool:
        return self.system == "darwin"

    @property
    def is_apple_silicon(self) -> bool:
        return self.is_macos and self.machine == "arm64"


@dataclass(frozen=True)
class CheckpointKind:
    """What a checkpoint directory holds, from its ``config.json``."""

    format: Literal["pytorch", "mlx"]
    weights: Literal["bf16", "8bit", "mixed"]


def checkpoint_kind(config: dict[str, Any], directory: Path | str = "checkpoint") -> CheckpointKind:
    """Classify a parsed ``config.json``; ``ValueError`` carries the refusal message.

    ``directory`` is only there so the message can name the checkpoint the user passed.
    """
    model_type = config.get("model_type")
    if model_type == _PYTORCH_MODEL_TYPE:
        return CheckpointKind("pytorch", "bf16")
    if model_type != _MLX_MODEL_TYPE:
        raise ValueError(
            f"{directory} has model_type {model_type!r}; expected {_PYTORCH_MODEL_TYPE!r} "
            f"(PyTorch) or {_MLX_MODEL_TYPE!r} (MLX)"
        )
    quantization = config.get("quantization")
    if quantization is None:
        return CheckpointKind("mlx", "bf16")
    # A hand-edited config could hold anything here; refuse it like any other unsupported
    # quantization rather than crash the launch with a traceback.
    if not isinstance(quantization, dict):
        quantization = {}
    if quantization.get("bits") == 8 and quantization.get("mode") == "mxfp8":
        return CheckpointKind("mlx", "8bit")
    # scripts/make_mixed_checkpoint.py: bf16 with the depth decoder's linear layers in affine int8
    # (mlx-audio quantizes exactly the layers whose weights come with scales).
    if quantization.get("bits") == 8 and quantization.get("mode", "affine") == "affine":
        return CheckpointKind("mlx", "mixed")
    raise ValueError(
        f"{directory} is {quantization.get('bits')}-bit {quantization.get('mode')}; "
        "the MLX backend supports bf16, 8-bit (mxfp8) and mixed (affine 8-bit)"
    )


@dataclass(frozen=True)
class Settings:
    """Launch configuration. Every field is already validated.

    Every field is required: ``Settings`` is only ever built by
    ``settings_from_args``, which fills in every default and runs every
    check, so there is no partially-defaulted instance to accidentally ship.
    Tests that need a ``Settings`` build one through ``settings_from_args``
    too, not by constructing this dataclass directly.

    ``ws_port`` is ``None`` when the WebSocket server is disabled (either by
    ``--ws-port disabled`` or, going forward, any other way disabling is
    spelled); otherwise it is a validated port distinct from ``port``.

    ``cors`` is empty when CORS is off (the default); ``("*",)`` when any
    origin is allowed; otherwise a tuple of validated, deduplicated origins
    (see ``_parse_cors_origins``).
    """

    model_path: Path
    host: str
    port: int
    ws_port: int | None
    cors: tuple[str, ...]
    split_chars: int
    chunk_first: int
    chunk_max: int
    voices_dir: Path
    weights: Literal["bf16", "8bit", "mixed"]


# --port and --ws-port must be plain unsigned integers: no leading '+' or
# '-', no decimal point, no exponent, no surrounding whitespace, and no
# non-ASCII digit (e.g. full-width '８０８０' or Arabic-Indic '٨٠٨٠') -- int()
# itself accepts all of those, so the strict, ASCII-only shape is enforced
# separately. An explicit [0-9] class (rather than \d with re.ASCII) keeps
# that ASCII-only intent visible at the call site.
_STRICT_PORT_RE = re.compile(r"[0-9]+")


def _parse_port(value: str) -> int:
    """argparse ``type=`` for ``--port``: a plain, strictly-digit port number."""
    if not _STRICT_PORT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"--port must be a plain number (got {value!r})"
        )
    return int(value)


def _parse_ws_port(value: str) -> int | None:
    """argparse ``type=`` for ``--ws-port``: a plain digit port number, or ``disabled``."""
    if value.strip().lower() == "disabled":
        return None
    if not _STRICT_PORT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"--ws-port must be a port number or 'disabled' (got {value!r})"
        )
    return int(value)


def _parse_cors_origins(value: str | None) -> tuple[str, ...]:
    """``None`` (flag absent) disables CORS; ``"*"`` (bare flag) allows any origin.

    Otherwise ``value`` is a comma-separated list. Each entry is trimmed,
    validated and canonicalized as a bare origin (``canonical_origin``, in
    ``origins.py`` so ``cors.py``'s ``CorsMiddleware`` can canonicalize an
    incoming ``Origin`` header the same way before comparing) and deduplicated, preserving
    first-seen order -- so ``http://a.example:80`` and ``http://a.example``
    collapse to one entry. ``"*"`` means any origin only when every entry is
    ``"*"`` (so ``"*,*"`` collapses to ``("*",)``); mixed with any other
    origin it is rejected, as is an allowlist that is empty after trimming.
    """
    if value is None:
        return ()
    raw_entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not raw_entries:
        # An empty allowlist would silently mean "CORS off" despite the flag. No "--cors" here:
        # `settings_from_args` prepends the flag name once, uniformly, for every rejection this
        # function raises (review-agent final pass, issue 3).
        raise ValueError("empty origin list")
    if "*" in raw_entries:
        if any(entry != "*" for entry in raw_entries):
            raise ValueError("'*' can't be combined with other origins")
        return ("*",)
    origins: list[str] = []
    seen: set[str] = set()
    for entry in raw_entries:
        normalized = canonical_origin(entry)
        if normalized not in seen:
            seen.add(normalized)
            origins.append(normalized)
    return tuple(origins)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model_path", type=Path, help="Path to the model directory")
    parser.add_argument(
        "--backend",
        choices=("cuda", "mlx"),
        default="mlx",
        help="Accepted for compatibility: mlx is the only backend",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address for HTTP and WebSocket (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=DEFAULT_PORT,
        help=f"HTTP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--ws-port",
        type=_parse_ws_port,
        default=_WS_PORT_UNSET,
        metavar="PORT|disabled",
        help=(
            "WebSocket port (default: HTTP port + 1); pass 'disabled' to "
            "turn off the WebSocket server"
        ),
    )
    parser.add_argument(
        "--cors",
        nargs="?",
        const="*",
        default=None,
        metavar="ORIGINS",
        help=(
            "Enable CORS: bare flag allows any origin, or pass a "
            "comma-separated allowlist (default: disabled). Takes an "
            "optional value, so pass it after model_path, or use "
            "--cors=ORIGINS, to avoid swallowing the next argument"
        ),
    )
    parser.add_argument(
        "--split-chars",
        type=int,
        default=DEFAULT_SPLIT_CHARS,
        metavar="N",
        help=(
            "Split long text at sentence-ish boundaries near this many "
            f"characters; 0 disables splitting (default: {DEFAULT_SPLIT_CHARS})"
        ),
    )
    parser.add_argument(
        "--chunk-first",
        type=int,
        default=DEFAULT_CHUNK_FIRST,
        metavar="N",
        help=f"Codec frames in the first streamed flush (default: {DEFAULT_CHUNK_FIRST})",
    )
    parser.add_argument(
        "--chunk-max",
        type=int,
        default=DEFAULT_CHUNK_MAX,
        metavar="N",
        help=f"Largest flush the stream ramps up to, in codec frames (default: {DEFAULT_CHUNK_MAX})",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=DEFAULT_VOICES_DIR,
        help=f"Directory holding saved voices (default: ./{DEFAULT_VOICES_DIR})",
    )
    return parser


def _read_config(model_path: Path) -> dict[str, Any] | None:
    """The checkpoint's ``config.json``, or ``None`` when it is missing or not a JSON object."""
    try:
        config = json.loads((model_path / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _check_platform(
    parser: argparse.ArgumentParser, args: argparse.Namespace, platform: Platform | None
) -> None:
    """Refuse a machine the MLX backend can't serve (FR-002, FR-003, FR-004).

    ``platform`` is ``None`` for callers that only parse arguments (the model-free tests); the
    entry point passes ``Platform.detect()``, so the machine is always checked at launch.
    """
    if args.backend == "cuda":
        parser.error(CUDA_REMOVED)
    if platform is None:
        return
    if not platform.is_apple_silicon:
        parser.error(
            f"the server needs an Apple Silicon Mac (this machine: {platform.system} {platform.machine})"
        )
    if platform.memory_bytes < MLX_MIN_MEMORY_BYTES:
        gigabytes = round(platform.memory_bytes / 1024**3, 1)
        parser.error(f"the server needs at least 16 GB of memory (this machine: {gigabytes:g} GB)")


def _check_checkpoint(
    parser: argparse.ArgumentParser, model_path: Path
) -> Literal["bf16", "8bit", "mixed"]:
    """Refuse a checkpoint that isn't MLX; return its weight precision, read from the file."""
    config = _read_config(model_path)
    if config is None:
        parser.error(
            f"{model_path} has no readable config.json; the server needs MLX weights "
            f"({MLX_WEIGHTS_HINT})"
        )
    try:
        kind = checkpoint_kind(config, model_path)
    except ValueError as exc:
        parser.error(str(exc))
    if kind.format == "pytorch":
        parser.error(
            f"{model_path} is the PyTorch checkpoint; the server needs MLX weights "
            f"({MLX_WEIGHTS_HINT})"
        )
    return kind.weights


def settings_from_args(
    argv: Sequence[str] | None = None, platform: Platform | None = None
) -> Settings:
    """Parse ``argv`` (``sys.argv[1:]`` when ``None``) into a validated ``Settings``.

    Every rejection goes through ``parser.error()``, which prints the usage
    and message to stderr and raises ``SystemExit(2)`` -- the same failure
    mode as an argparse-level type error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.model_path.is_dir():
        parser.error(f"model_path must be an existing directory (got {args.model_path})")

    _check_platform(parser, args, platform)
    weights = _check_checkpoint(parser, args.model_path)

    port = args.port
    if not 1 <= port <= 65535:
        parser.error(f"--port must be between 1 and 65535 (got {port})")

    ws_port = args.ws_port
    ws_port_derived = ws_port is _WS_PORT_UNSET
    if ws_port_derived:
        ws_port = port + 1
    if ws_port is not None:
        if not 1 <= ws_port <= 65535:
            if ws_port_derived:
                parser.error(
                    f"the derived --ws-port (--port + 1 = {ws_port}) is out of "
                    "range; pass --ws-port to set it explicitly"
                )
            parser.error(
                f"--ws-port must be between 1 and 65535, or 'disabled' (got {ws_port})"
            )
        if ws_port == port:
            parser.error(f"--ws-port must differ from --port (got {ws_port})")

    if args.split_chars < 0:
        parser.error(f"--split-chars must be >= 0 (got {args.split_chars})")

    if args.chunk_first < 1:
        parser.error(f"--chunk-first must be at least 1 (got {args.chunk_first})")
    if args.chunk_max < 1:
        parser.error(f"--chunk-max must be at least 1 (got {args.chunk_max})")
    # main.cpp parity: the first flush never streams past the ramp ceiling.
    chunk_first = min(args.chunk_first, args.chunk_max)

    try:
        cors = _parse_cors_origins(args.cors)
    except ValueError as exc:
        # `_parse_cors_origins`/`canonical_origin` (origins.py) keep their own messages neutral
        # about which flag they're validating -- `canonical_origin` is also run on every request's
        # incoming `Origin` header, where "--cors" would be meaningless -- so this is the one place
        # that actually knows the value came from `--cors` (review-agent final pass, issue 3).
        parser.error(f"--cors: {exc}")

    return Settings(
        model_path=args.model_path,
        host=args.host,
        port=port,
        ws_port=ws_port,
        cors=cors,
        split_chars=args.split_chars,
        chunk_first=chunk_first,
        chunk_max=args.chunk_max,
        voices_dir=args.voices_dir,
        weights=weights,
    )
