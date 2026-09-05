# Breeze TTS on Apple Silicon

MPS inference, incremental PCM streaming, and experimental MLX int8 decoding for
[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2).

## Install

Apple Silicon Mac, native arm64 Python 3.11–3.13, and an MPS-enabled PyTorch build.
Run commands from the repository root in the activated environment.

```bash
git clone https://github.com/mgwilt/breeze-tts-mps.git
cd breeze-tts-mps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable'"
```

Download the complete [model checkpoint](https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/main)
to `models/Breeze-TTS-2`, or substitute its path below.

| Dependency | Version |
| --- | --- |
| PyTorch / torchaudio | 2.9.1 |
| Transformers | 4.57.3 |
| Qwen codec package (`qwen-tts`) | 0.1.1 |
| MLX / MLX Metal, optional | 0.32.0 |

## Serve with MPS

```bash
python -m breeze_infer.api models/Breeze-TTS-2 \
  --device mps --host 127.0.0.1 --port 7860
```

In another terminal:

```bash
curl --fail http://127.0.0.1:7860/health

curl --fail --no-buffer http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=Welcome aboard. Your journey begins now.' \
  -F 'instruction=A warm, clear voice with calm, thoughtful delivery.' \
  -F 'cfg_scale=4' \
  -F 'seed=42' \
  --output speech.pcm
```

| API behavior | Contract |
| --- | --- |
| Output | Raw mono PCM, 24 kHz, signed 16-bit little-endian; not WAV |
| Streaming | Incremental chunks while generation runs |
| Concurrency | One synthesis request; overlapping requests return HTTP 409 |
| Cancellation | Client disconnect cancels generation; cleanup precedes lock release |
| Input bounds | Nonempty text ≤4,000 characters; nonempty instruction ≤2,000; token-capacity checks also apply |
| Reference audio, MPS | Supply both `ref_audio=@reference.wav` and its exact `ref_text` transcript |

Keep inference on loopback. Network exposure requires a separate authenticated TLS edge.

## Serve with experimental MLX int8

```bash
python -m pip install mlx==0.32.0 mlx-metal==0.32.0
```

Stop the MPS process before reusing its port. Recipe selection currently uses the internal
Python configuration seam; the API CLI does **not** expose an MLX flag.

```bash
python - <<'PY'
from pathlib import Path
import uvicorn
from breeze_infer import api

api._settings = api.ApiSettings(
    model=Path("models/Breeze-TTS-2"),
    device="mps",
    experimental_recipe="mlx-int8-v1",
    fast_all=False,
    fast_text_encoder=False,
    fast_backbone_prefill=False,
    fast_backbone_decode=False,
    fast_depth_decoder=False,
    fast_codec=False,
)
uvicorn.run(api.app, host="127.0.0.1", port=7860)
PY
```

Use the same request above. Requirements: CFG **4**, a uint32 seed, and instruction-only
voice design. Reference-audio cloning/direction is not supported by this recipe.
Dependency versions and Metal kernel hashes are checked by
[`experimental.py`](breeze_infer/experimental.py).

`--fast-all` and the individual `--fast-*` flags control upstream **CUDA** optimizations,
not Apple Silicon acceleration. MLX sampling is not Torch-seed equivalent; int8 remains
experimental, not perceptually release-accepted.

## What this fork changes

| Component | Apple Silicon implementation |
| --- | --- |
| Runtime | Portable PyTorch MPS/CPU path; CUDA path retained |
| Streaming codec | Stateful incremental decoding, bounded producer queue, cancellation and failure cleanup |
| PyTorch depth decoding | Cached prefixes and batched conditional/unconditional CFG |
| MLX generation | Backbone and depth decoder ports; compiled decode steps, SDPA, branch-isolated KV caches |
| Hybrid boundary | PyTorch BF16 text preparation/prefill → MLX generation → PyTorch FP32 codec |
| Int8 recipe | Affine weight-only quantization, group size 64; 196 backbone + 84 depth linear weights |
| Unquantized components | Embeddings, norms, projectors, custom/output heads, codec; BF16 MLX activations and KV |
| Memory scope | Original Torch weights remain resident; this is not an end-to-end MLX rewrite |

## Tests

```bash
python -m pytest tests -q
```

Install the optional MLX dependencies to exercise MLX tests on Metal. Tests cover cached
decoding, CFG isolation, sampling, codec boundaries, cancellation, and failure recovery.
Mac results do not establish CUDA correctness or perceptual equivalence.

## Recorded performance

Apple M3 Ultra, 512 GiB unified memory; September 2026. Checkpoint revision
`799624c0b4a1daa8db6d28bbd9850043c0270734`; dependency versions listed above.
Warm, uncached synthesis at CFG 4, temperature 0.9, top-k 50, top-p 1.0.

### Implementation progression

| Implementation | Timed / warmups | p95 total RTF | p95 first PCM |
| --- | ---: | ---: | ---: |
| Original buffered MPS reference | 10 / 3 | 9.954 | 49.165 s |
| Cached depth + incremental codec | 3 / 1 | 7.013 | 0.809 s |
| MPS SDPA + direct output-head indexing | 30 / 3 | 3.442 | 0.665 s |
| MLX int8 backbone + depth | 30 / 3 | 0.799 | 0.393 s |

Historical cohorts, **not a controlled speedup ablation**: sample counts, generated
durations, execution and first-PCM observation boundaries differ. The first two runs
have no recorded wall-clock date; order denotes implementation milestones. SDPA and MLX
share ten prompts × three seeds, but use different RNG streams. The reference emits PCM
only after completion; its first-PCM reduction includes incremental delivery.

### Matched MLX weight-precision study

Same 18 prompt/instruction/seed cases and three warmups per arm; no CFG reduction.

| Backbone / depth weights | p95 steady RTF | p95 first PCM | ASR word edits / 189 |
| --- | ---: | ---: | ---: |
| BF16 / BF16 | 1.088 | 0.431 s | 3 |
| Int8 / BF16 | 1.008 | 0.422 s | 5 |
| BF16 / Int8 | 0.770 | 0.400 s | 6 |
| Int8 / Int8 | 0.688 | 0.390 s | 7 |

Both-int8 reduces p95 steady RTF **36.7%** versus BF16 weights. Selected linear-weight
storage, including scales/biases: **3,485,466,624 → 1,851,654,144 bytes (−46.875%)**.
This is not total model size or peak process memory.

- Total RTF = request wall time / generated audio duration; **<1 is faster than real time**.
- Steady RTF excludes time and audio through the first chunk. p95 uses nearest rank,
  excluding warmups; at n=18 it is the maximum.
- Output durations differ even in the matched study. ASR edits are unadjudicated
  recognizer flags, not listening scores or proof of equal quality.
- First PCM is not browser or acoustic playback latency. Smaller Macs and
  concurrent-model inference are not characterized by these measurements.
- These are retained historical measurements, not a benchmark rerun of the current
  checkout. Raw benchmark receipts and audio are not bundled in this repository.

## Upstream and license

Based on [BreezeBlue's Breeze TTS](https://github.com/breezeblue-ai/breeze-tts).
BreezeBlue authors the model; this community fork maintains the Apple Silicon changes.
The audio tokenizer is based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

Source: [Apache 2.0](LICENSE). Model weights, derivatives, and self-hosted outputs:
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
The code license does not grant commercial model rights. Obtain consent and rights for
reference voices; unauthorized cloning, impersonation, and fraud are prohibited.
