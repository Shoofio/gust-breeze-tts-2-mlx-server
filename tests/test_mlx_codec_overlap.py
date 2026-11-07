"""mlx-audio's streaming codec decode matches a one-shot decode at any chunk size, on the stock code
(the fix is Blaizzy/mlx-audio#1003), and `require_codec_fix` refuses an mlx-audio without it.

The decode tests are model-free but need mlx and mlx-audio (Apple Silicon), skipped elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.mlx_streaming import require_codec_fix

CHUNKINGS = ([1] * 12, [2] * 6, [3, 1, 4, 4], [12])


@pytest.mark.parametrize("version", ["0.5.9", "0.6.4", "0.10.0", "1.0.0", "0.6.0.dev3"])
def test_versions_with_the_fix_are_accepted(version: str) -> None:
    require_codec_fix(version)


@pytest.mark.parametrize("version", ["0.5.7", "0.5.8", "0.4.2"])
def test_versions_without_the_fix_are_refused(version: str) -> None:
    with pytest.raises(RuntimeError, match=r"predates the streaming codec fix \(Blaizzy/mlx-audio#1003\)"):
        require_codec_fix(version)


def test_the_installed_mlx_audio_has_the_fix() -> None:
    pytest.importorskip("mlx_audio")
    require_codec_fix()


def _block(upsample_rate: int):
    mx = pytest.importorskip("mlx.core")
    speech_tokenizer = pytest.importorskip("mlx_audio.tts.models.qwen3_tts.speech_tokenizer")
    mx.random.seed(0)
    block = speech_tokenizer.DecoderBlockUpsample(in_dim=6, out_dim=5, upsample_rate=upsample_rate)
    # A non-zero bias: a zero bias hides a doubled one.
    block.conv.bias = mx.random.normal(block.conv.bias.shape)
    return block


def _streamed(block, x, chunk_sizes):
    import mlx.core as mx

    block.reset_state()
    out, start = [], 0
    for size in chunk_sizes:
        out.append(block.step(x[:, start : start + size, :]))
        start += size
    return mx.concatenate(out, axis=1)


@pytest.mark.parametrize("upsample_rate", [2, 4, 8])
@pytest.mark.parametrize("chunk_sizes", CHUNKINGS, ids=lambda c: "-".join(map(str, c)))
def test_streamed_decode_matches_one_call(upsample_rate, chunk_sizes):
    block = _block(upsample_rate)
    import mlx.core as mx

    x = mx.random.normal((1, 12, 6))
    np.testing.assert_allclose(np.array(_streamed(block, x, chunk_sizes)), np.array(block(x)), atol=1e-5)
