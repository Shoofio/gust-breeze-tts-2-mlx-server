# Gust: real-time Breeze TTS 2 on Apple Silicon

> [!IMPORTANT]
> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only.**
>
> An unofficial, independent project, not affiliated with or endorsed by BreezeBlue. The code is Apache 2.0. The model weights, derivative models and the audio you generate with them are under the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). See [License and responsible use](#license-and-responsible-use).

A local text-to-speech server for Macs with Apple Silicon. You send it text over HTTP or a WebSocket, and it streams the speech back while it is still generating, fast enough to talk with: the first audio arrives in about **140 ms** on an M4 Pro, and the rest is generated faster than it plays.

It runs [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), an open-weight speech model by BreezeBlue, on the Mac's GPU through [MLX](https://github.com/ml-explore/mlx). BreezeBlue's own code streams only on NVIDIA GPUs; on a Mac it runs more than twice slower than real time and can't stream. This server makes the Mac a real-time option, with the model's full feature set.

## What it can do

- **Clone a voice** from a short recording (up to 30 s) and its transcript.
- **Direct the delivery** in plain words: *"Speak playfully, with a teasing, amused lilt"*, *"Whisper softly"*, *"Restrained and matter-of-fact"*.
- **Design a voice** from a description alone, with no recording: *"A gravelly, elderly man with a warm storyteller's voice"*.
- **Audio tags** in the text, such as `(laughs)`, `(sighs)` and `(chuckles)` ([BreezeBlue's list](https://docs.breezeblue.ai/guides/audio-tags)).
- **Save voices** once and reuse them by name (`/v1/voices`).
- **Play in a browser**: `GET /v1/audio/speech.wav?text=...` is a WAV stream an `<audio>` element can play as it arrives.
- **Stream raw audio** over HTTP or WebSocket (mono, 24 kHz, 16-bit PCM), with the same API as the [Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) server.
- English and Chinese, the languages Breeze TTS 2 supports.

## How fast

Apple M4 Pro (14-core CPU, 20-core GPU, 48 GB). "First audio" is the time to the first sound, including any buffer needed so playback never stalls. A real-time factor (RTF) below 1 means audio is produced faster than it plays.

| Build | First audio, plain / with a direction | RTF, with a direction | Memory |
| --- | --- | --- | --- |
| BreezeBlue's PyTorch code on the Mac GPU | no streaming: ~6 s for one sentence | 2.26 (plain lines) | 11 GB |
| `8bit` (mlx-community conversion) | 212 / 297 ms | 0.58 | 13 GB |
| `bf16` (mlx-community conversion) | 380 / 456 ms | 0.86 | 15 GB |
| **Gust (the default; `mixed` builds the same model)** | **139 / 206 ms** | **0.62** | **8.9 GB** |
| *For scale: RTX 3090, power-capped at 250 W, BreezeBlue's CUDA fast path* | *87 / 160 ms* | *0.50* | *19.7 GB VRAM* |

Measured with `scripts/bench_mac.py`: 24 lines of one to four [Harvard sentences](https://www.cs.columbia.edu/~hgs/audio/harvard.html), two in three with a voice direction, cloning a 24 s LibriSpeech voice (speaker 1272, CC BY 4.0). Results on other Macs are welcome in the issues.

## Requirements

- A Mac with Apple Silicon (M1 or later) and at least 16 GB of memory. The server refuses to start on an Intel Mac or with less memory.
- macOS 14 (Sonoma) or later.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the dependencies).
- About 7 GB of disk for the model.

## Quick start

```bash
git clone https://github.com/Shoofio/gust-breeze-tts-2-mlx-server.git
cd gust-breeze-tts-2-mlx-server

# the model (~7 GB, once)
uvx --from huggingface_hub hf download Shoof/Gust-Breeze-TTS-2-MLX-Realtime \
  --revision 928d97bb7404a6b7d2b68dbea0e08cdff4712450

# the server, on http://127.0.0.1:8080
scripts/start_breeze_mac.sh --host 127.0.0.1
```

The first start also creates `.venv` and installs the dependencies, so it takes longer than later ones. If the model is missing, the launcher prints the exact download command and exits. When `curl http://127.0.0.1:8080/health` answers `{"status":"ok",...}`, open this in a browser:

```text
http://127.0.0.1:8080/v1/audio/speech.wav?text=Hello%20there.%20This%20is%20running%20on%20my%20Mac.
```

> [!WARNING]
> Without `--host 127.0.0.1` the server listens on every network interface with CORS open to any site, so anyone on your network can use it. Keep `--host 127.0.0.1` unless you mean to share it.

## Using it

### Design a voice

With no recording, the instruction describes the voice. A higher `cfg_scale` (2 to 4) follows the description more closely:

```bash
curl -G http://127.0.0.1:8080/v1/audio/speech.wav \
  --data-urlencode "text=The birch canoe slid on the smooth planks." \
  --data-urlencode "instruction=A deep, calm radio announcer with crisp, precise diction." \
  --data-urlencode "cfg_scale=2" \
  --output announcer.wav
```

### Clone a voice and save it

Register a recording (WAV, FLAC or OGG, up to 30 s) with its exact transcript, once:

```bash
curl -X POST http://127.0.0.1:8080/v1/voices \
  -F "name=narrator" \
  -F "ref_audio=@narrator.wav" \
  -F "ref_text=The exact words spoken in the recording."
```

Then speak in that voice, directed, with an audio tag:

```bash
curl -G http://127.0.0.1:8080/v1/audio/speech.wav \
  --data-urlencode "voice_id=narrator" \
  --data-urlencode "text=It's easy to tell the depth of a well. (laughs) These days a chicken leg is a rare dish." \
  --data-urlencode "instruction=Speak playfully, amused." \
  --data-urlencode "cfg_scale=2" \
  --output narrator.wav
```

Named voices are saved under `./voices` and load again at the next start.

### Stream into your own app

- `POST /v1/audio/speech` takes the same fields as a form (and can take `ref_audio` directly, without saving a voice) and streams raw PCM: mono, 24 kHz, signed 16-bit little-endian.
- The WebSocket API (port 8081 by default) streams PCM chunks with progress events, for many lines over one connection.
- One line is generated at a time; a second `POST` while one is running gets `409 busy` (the `.wav` route waits up to 60 s instead).

Every route, field, limit and error is in [docs/api.md](docs/api.md).

## Precision

`scripts/start_breeze_mac.sh [--precision mixed|8bit|bf16]`:

- **No flag (the default)**: the [Gust checkpoint](https://huggingface.co/Shoof/Gust-Breeze-TTS-2-MLX-Realtime), the real-time build. The backbone stays bf16, exactly as trained, and only the depth decoder's linear layers are int8.
- **`mixed`**: the same model, built at load from mlx-community's bf16 conversion, with the same audio. Use it if you already have the bf16 weights.
- **`8bit`**: mlx-community's mxfp8 conversion. Fast, but it quantizes the backbone too, and it sounded worst in the listening test below. It measured 7 GB on a 16 GB Mac.
- **`bf16`**: the full-precision reference. Slower than real time on smaller Macs.

Each is pinned to a revision, and the launcher prints the download command for whichever weights are missing.

## How it got fast

Generation is memory-bound: for every audio frame (80 ms of sound), the model reads its backbone once and its depth decoder 15 times, so the depth decoder is about three quarters of the bytes read per frame. The default build uses four settings, each also usable on its own (the Gust checkpoint has the first already applied):

| Setting | What it does |
| --- | --- |
| `BREEZE_MLX_QUANT=depth:8` | int8 for the depth decoder only. The backbone, which carries prosody and where a voice direction lands, stays exact. |
| `BREEZE_MLX_COMPILE=1` | Runs each frame's sampling and its 15 depth-decoder steps as one compiled MLX graph. |
| `BREEZE_MLX_FAST_FIRST=6` | Sends the first six frames one at a time, so the first audio doesn't wait for a chunk to fill. |
| `BREEZE_MLX_CACHE_GB=1` | Caps MLX's buffer cache, which otherwise holds ~6 GB of freed memory. |

## A fix in mlx-audio

Along the way we found a streaming bug in [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s Qwen3-TTS codec decoder, which Breeze TTS 2 uses: the transposed convolutions' bias was added twice at every chunk boundary, so streamed audio depended on the chunk size, worst in an utterance's first frames. We contributed the fix upstream as [Blaizzy/mlx-audio#1003](https://github.com/Blaizzy/mlx-audio/pull/1003) (merged). It shipped in mlx-audio 0.5.9; this server pins 0.6.4, refuses to start on an older mlx-audio, and `tests/test_mlx_codec_overlap.py` checks that streamed decode is chunk-size independent on the stock code. If you stream Breeze TTS 2 with mlx-audio yourself, use 0.5.9 or later.

## Quality

The same tokens through each build and through bf16, comparing the samplers' logits (KL divergence, lower is closer; lines with a voice direction):

| Build | Backbone | Depth decoder |
| --- | --- | --- |
| `8bit` (mxfp8) | 0.0030 | 0.0067 |
| `mixed` | 0 (exact) | 0.0020 |

Voice directions amplify quantization error, because classifier-free guidance multiplies the gap between its two passes; `8bit` quantizes the backbone, `mixed` doesn't. In a blind listening test (one listener, ten lines, four builds), `8bit` was judged worst on seven lines and best on none, while `bf16`, `mixed` and an all-int8 build tied.

Side-by-side samples, an RTX 3090 against Gust on an M4 Pro: [the Gust model card](https://huggingface.co/Shoof/Gust-Breeze-TTS-2-MLX-Realtime#samples).

## Tools

| Script | What it does |
| --- | --- |
| `scripts/bench_mac.py` | The speed benchmark above, against a running server. |
| `scripts/mlx_fidelity.py` | The KL comparison of a build against bf16. |
| `scripts/make_mixed_checkpoint.py` | Builds the `mixed` checkpoint from the bf16 one and verifies it tensor by tensor. |

## Development

The launcher creates `.venv` on its first run.

| Action | Command |
| --- | --- |
| Unit and integration tests (no model) | `.venv/bin/pytest` |
| MLX tests (loads the model) | `BREEZE_MLX_MODEL=<model dir> .venv/bin/pytest -m mlx` |
| Lint | `.venv/bin/ruff check .` |
| HTTP API benchmark | `.venv/bin/python -m breeze_infer.bench_api --url http://127.0.0.1:8080` |
| SillyTavern live test | `node tests/live/sillytavern/run.mjs <health\|voices\|speech\|full>` |
| C++ server docs example check | `.venv/bin/python -m tests.live.cpp_examples --url http://127.0.0.1:8080` |

`specs/` holds the design history of the API contracts and the MLX backend; [CHANGELOG.md](CHANGELOG.md) lists every change.

## Credits

- [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) and its inference code by [BreezeBlue](https://breezeblue.ai).
- The streaming server this grew from: [northcraftfoundries/breeze-tts2-fast-streaming-api](https://github.com/northcraftfoundries/breeze-tts2-fast-streaming-api), which also serves NVIDIA GPUs. For NVIDIA, use it or [BreezeBlue's own code](https://github.com/breezeblue-ai/breeze-tts).
- The MLX model in [mlx-audio](https://github.com/Blaizzy/mlx-audio) and the [mlx-community](https://huggingface.co/mlx-community) conversions.
- The audio tokenizer, based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team (Apache 2.0).

Assisted by Claude Opus 5.5.

## License and responsible use

The source code is under the [Apache License, Version 2.0](LICENSE), both the upstream code and this project's changes.

Model weights, checkpoints, derivative models and self-hosted outputs are governed separately by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE): research and non-commercial use only. The Apache license here grants no right to use the model or its outputs commercially, and running this server doesn't change that. Outputs from BreezeBlue's own hosted platform at [breezeblue.ai](https://breezeblue.ai/) follow its [Terms of Service](https://breezeblue.ai/legal/terms) instead.

You are responsible for having the rights and consent for every voice you clone and every input you use. Unauthorized voice cloning, impersonation, fraud and other unlawful or harmful uses are prohibited.

The code and model materials are provided "as is", without warranties, and without liability to the extent permitted by law. Third-party components remain under their own licenses.
