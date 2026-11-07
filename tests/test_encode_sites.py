"""Guard against a new direct call to the audio codec/tokenizer's ``.encode(...)``
outside ``breeze_infer/audio.py``.

Every reference-audio encode in the serving path goes through ``encode_prompt_waveform``,
the one place that downmixes the waveform and checks and normalizes the codes it returns
(``int16[frames, codebooks]``, on the CPU). It was introduced to pin cuDNN for the encode on
the CUDA backend (research.md R18); the single entry point is kept so a new call site can't
skip those checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIO_PY = REPO_ROOT / "breeze_infer" / "audio.py"

# The real call shape (checked by hand, not guessed): `encoded =
# audio_tokenizer.encode(wav, sr=...)` in breeze_infer/audio.py. Matching on the receiver
# containing "tokenizer" or "codec_model" -- not a bare `.encode(` regex -- so this doesn't
# flag the many unrelated `str.encode("utf-8")`/`.encode("ascii")` calls elsewhere
# (breeze_infer/cors.py, version_header.py, ...). `MlxAudioTokenizer.encode` in
# models/mlx_streaming.py is the tokenizer itself (its `self._codec.encode` is the codec
# underneath), not a call site.
_RECEIVER_MARKERS = ("tokenizer", "codec_model")


class _EncodeCallFinder(ast.NodeVisitor):
    """Records (enclosing function name or None, line number) for every call that
    looks like a codec/tokenizer `.encode(...)`."""

    def __init__(self) -> None:
        self._func_stack: list[str] = []
        self.hits: list[tuple[str | None, int]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "encode":
            receiver = ast.unparse(func.value)
            if any(marker in receiver for marker in _RECEIVER_MARKERS):
                enclosing = self._func_stack[-1] if self._func_stack else None
                self.hits.append((enclosing, node.lineno))
        self.generic_visit(node)


def _production_python_files() -> list[Path]:
    files = sorted((REPO_ROOT / "breeze_infer").rglob("*.py"))
    files += sorted((REPO_ROOT / "models").rglob("*.py"))
    return files


def _find_offending_calls(path: Path, source: str) -> list[str]:
    """The reusable check: given a file's source, return one description string per
    codec/tokenizer `.encode(...)` call that isn't allowed at that path. Separated from
    file I/O so it can be exercised directly against a synthetic source below.
    """
    tree = ast.parse(source, filename=str(path))
    finder = _EncodeCallFinder()
    finder.visit(tree)

    offenders = []
    for func_name, lineno in finder.hits:
        offenders.append(f"{path}:{lineno} (in {func_name or '<module>'})")
    return offenders


def test_detects_a_direct_codec_encode_call_outside_audio_py() -> None:
    """Proves the detector itself works, independent of this repo's current (clean)
    state: a direct `self.codec_model.encode(...)` call in a file that isn't
    breeze_infer/audio.py must be reported.
    """
    decoy_path = REPO_ROOT / "models" / "decoy.py"
    source = (
        "class Thing:\n"
        "    def some_other_method(self, batch):\n"
        "        return self.codec_model.encode(batch)\n"
    )

    offenders = _find_offending_calls(decoy_path, source)

    assert offenders == [f"{decoy_path}:3 (in some_other_method)"]


def test_only_audio_py_calls_the_codec_encode_directly() -> None:
    offenders: list[str] = []
    for path in _production_python_files():
        if path == _AUDIO_PY:
            continue
        offenders.extend(_find_offending_calls(path, path.read_text()))

    assert not offenders, (
        "direct codec/tokenizer .encode(...) call(s) found outside "
        "breeze_infer/audio.py: "
        + "; ".join(offenders)
        + " -- route reference-audio encoding through "
        "breeze_infer.audio.encode_prompt_waveform instead."
    )
