#!/bin/sh
# Starts the Breeze TTS 2 streaming server on an Apple Silicon Mac.
#
# Usage: scripts/start_breeze_mac.sh [--precision mixed|8bit|bf16] [server options...]
#
# Without --precision: the ready-made mixed checkpoint (MIXED_REPO below), the real-time build:
#   the bf16 backbone with the depth decoder's linear layers in int8, made by
#   scripts/make_mixed_checkpoint.py.
# --precision mixed: the same model, built at load from the bf16 conversion (BF16_REPO): its
#   depth decoder is quantized to int8 when the server starts. The audio is the same as the
#   ready-made checkpoint's; use it if you already have the bf16 weights.
# --precision 8bit / bf16: the mlx-community conversions as they are (8-bit mxfp8, or bf16).
#
# The mixed builds also compile the frame, send the first frames one by one, cap MLX's buffer
# cache at 1 GB and stream one-frame chunks; each BREEZE_MLX_* setting can be overridden from
# the environment (docs/api.md "Speed settings"; models/mlx_streaming.py, MlxSpeedOptions).

# Plain sh (see shebang), so no pipefail -- it isn't defined outside bash/zsh.
set -eu

# The checkpoints, each pinned to a revision so a re-upload can't change the weights under a
# running setup. The only place these repository ids live.
MIXED_REPO="Shoof/Gust-MLX"
MIXED_REVISION="928d97bb7404a6b7d2b68dbea0e08cdff4712450"
BF16_REPO="mlx-community/Breeze-TTS-2-mlx"
BF16_REVISION="3c8829fb7fd335818f085cd2ef49b4100c0e46c8"
Q8_REPO="mlx-community/Breeze-TTS-2-mlx-8bit"
Q8_REVISION="c6e4a2ff6ab9afba68b7853de802273ffe23fb49"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "start_breeze_mac.sh needs an Apple Silicon Mac (this machine: $(uname -s) $(uname -m))" >&2
    exit 1
fi

precision=default
if [ "${1:-}" = "--precision" ]; then
    if [ $# -lt 2 ]; then
        echo "--precision needs a value: mixed, 8bit or bf16" >&2
        exit 1
    fi
    precision="$2"
    shift 2
fi

# The speed settings of the mixed builds, unless the environment already sets them.
mixed_speed() {
    export BREEZE_MLX_COMPILE="${BREEZE_MLX_COMPILE-1}"
    export BREEZE_MLX_FAST_FIRST="${BREEZE_MLX_FAST_FIRST-6}"
    export BREEZE_MLX_CACHE_GB="${BREEZE_MLX_CACHE_GB-1}"
    CHUNKS="--chunk-first 1 --chunk-max 1"
}

case "$precision" in
    default)
        # Already quantized: no BREEZE_MLX_QUANT.
        REPO="$MIXED_REPO"
        REVISION="$MIXED_REVISION"
        mixed_speed
        ;;
    mixed)
        REPO="$BF16_REPO"
        REVISION="$BF16_REVISION"
        export BREEZE_MLX_QUANT="${BREEZE_MLX_QUANT-depth:8}"
        mixed_speed
        ;;
    8bit)
        REPO="$Q8_REPO"
        REVISION="$Q8_REVISION"
        ;;
    bf16)
        REPO="$BF16_REPO"
        REVISION="$BF16_REVISION"
        ;;
    *)
        echo "--precision must be mixed, 8bit or bf16 (got $precision)" >&2
        exit 1
        ;;
esac

# breeze_infer and models are imported as top-level packages, so the server
# must run from the repo root, regardless of the caller's working directory.
# Relative paths in "$@" therefore resolve against the repo root.
cd "$(dirname "$0")/.."

# mlx-audio declares newer transformers/huggingface_hub than the server pins,
# so a plain install fails; the overrides file keeps the server's pins
# (research R2). Only runs when the environment is missing or lacks mlx.
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import mlx.core, mlx_audio" 2>/dev/null; then
    if [ ! -x .venv/bin/python ]; then
        uv venv --python 3.12
    fi
    uv pip install -r requirements.txt --overrides requirements-mac-overrides.txt
fi

# HF_HOME is HuggingFace's own variable for relocating its cache; fall back to
# the default cache location when it is unset.
HUB_DIR="models--$(printf '%s' "$REPO" | sed 's|/|--|')"
MODEL="${HF_HOME:-$HOME/.cache/huggingface}/hub/$HUB_DIR/snapshots/$REVISION"

# A snapshot directory can exist with only some files (a partial download), so look for the
# weights index too.
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
    echo "Model snapshot not found or incomplete: $MODEL (set HF_HOME if the model cache lives elsewhere)" >&2
    echo "Download it with:" >&2
    echo "  uvx --from huggingface_hub hf download $REPO --revision $REVISION" >&2
    exit 1
fi

# exec replaces this shell with `uv run`, which runs the server as its child
# and forwards SIGTERM/SIGINT to it, so stopping this process stops the server.
# (To suspend or inspect the server itself, target the .venv python child.)
# Extra arguments pass straight through; a later --host or --cors wins.
#
# The launcher binds every interface (0.0.0.0) and enables CORS for every
# origin, so browser clients such as SillyTavern work without extra flags.
# That also turns off the server's cross-site 403 guard: any device on the
# network and any web page can upload or delete voices and run synthesis.
# --host 127.0.0.1 keeps it local; --cors=http://127.0.0.1:8000 narrows CORS.
# shellcheck disable=SC2086  # CHUNKS is two flags, or empty
exec uv run python -m breeze_infer.api "$MODEL" \
    --host 0.0.0.0 --port 8080 --cors '*' ${CHUNKS:-} \
    "$@"
