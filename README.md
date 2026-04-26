# NeuTTS

HuggingFace:

- NeuTTS-Air (English): [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf), [Space](https://huggingface.co/spaces/neuphonic/neutts-air)

- NeuTTS-Nano Multilingual Collection:
   - NeuTTS-Nano (English): [Model](https://huggingface.co/neuphonic/neutts-nano), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-q4-gguf)
   - NeuTTS-Nano-French: [Model](https://huggingface.co/neuphonic/neutts-nano-french), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-french-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-french-q4-gguf)
   - NeuTTS-Nano-German: [Model](https://huggingface.co/neuphonic/neutts-nano-german), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-german-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-german-q8-gguf)
   - NeuTTS-Nano-Spanish: [Model](https://huggingface.co/neuphonic/neutts-nano-spanish), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-spanish-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-spanish-q4-gguf)
   - [Multilingual Space](https://huggingface.co/spaces/neuphonic/neutts-nano-multilingual-collection)

[NeuTTS-Nano Demo Video](https://github.com/user-attachments/assets/629ec5b2-4818-4fa6-987a-99fcbadc56bc)

_Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI_

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS is a collection of open source, on-device TTS speech language models with instant voice cloning. Built off of LLM backbones, NeuTTS brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device—unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- **Best-in-class realism** — Natural, ultra-realistic voices at the sweet spot between speed, size, and quality
- **Optimised for on-device** — Quantisations in GGUF format, ready to run on phones, laptops, Raspberry Pis, and Apple Silicon Macs with GPU acceleration
- **Instant voice cloning** — Create your own speaker with as little as 3 seconds of audio
- **Simple LM + codec** — Straightforward architecture makes development and deployment simple
- **Apple Silicon GPU support** — Metal-accelerated inference on M-series Macs (M1–M4) with native one-command `run.sh` launcher
- **Tunable generation** — Expose sampling parameters (temperature, top-k) to control output diversity and quality
- **Performance optimized** — Cached token lookups, compiled regex, optimized streaming pipeline with reduced watermark overhead

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affiliated with Neuphonic, our github or this repo.
> We are on neuphonic.com only. Please be careful out there!

## Model Details

NeuTTS models are built from small LLM backbones—lightweight yet capable language models optimised for text understanding and generation—combined with powerful technologies designed for efficiency and quality:

- **Supported Languages**: English, Spanish, German, French (model-dependent)
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) — 50Hz neural audio codec with exceptional quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Quantisations available in GGUF format for efficient on-device inference
- **Responsibility**: Watermarked outputs for authenticity verification
- **Inference Speed**: Real-time generation on mid-range devices; GPU acceleration available
- **Power Consumption**: Optimised for mobile and embedded devices

|  | NeuTTS-Air | NeuTTS-Nano Models |
|---|---:|---:|
| **# Params (Active)** | ~360m | ~120m |
| **# Params (Emb + Active)** | ~552m | ~229m |
| **Cloning** | Yes | Yes |
| **License** | Apache 2.0 | NeuTTS Open License 1.0 |

## Throughput Benchmarking

Benchmarks for Q4_0 quantisations: [neutts-air-Q4_0](https://huggingface.co/neuphonic/neutts-air-q4-gguf) and [neutts-nano-Q4_0](https://huggingface.co/neuphonic/neutts-nano-q4-gguf). All models in the NeuTTS-Nano Multilingual Collection have identical architecture, so results apply across languages.

CPU benchmarking used [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) to measure prefill and decode throughput. GPU benchmarking (RTX 4090) uses vLLM to maximise throughput.

**Devices tested**: Galaxy A25 5G (CPU), AMD Ryzen 9 HX 370 (CPU), iMac M4 16GB (CPU + GPU estimates), Apple M2 16GB (CPU + GPU estimated), NVIDIA RTX 4090 (GPU)

|  | NeuTTS-Air | NeuTTS-Nano |
|---|---:|---:|
| **Galaxy A25 5G (CPU)** | 20 tokens/s | 45 tokens/s |
| **AMD Ryzen 9 HX 370 (CPU)** | 119 tokens/s | 221 tokens/s |
| **iMac M4 16GB (CPU)** | 111 tokens/s | 195 tokens/s |
| **Apple M2 16GB (GPU, Metal)** | ~300–500 tokens/s | ~400–700 tokens/s |
| **NVIDIA RTX 4090 (CUDA)** | 16,194 tokens/s | 19,268 tokens/s |

> [!NOTE]
> **CPU Benchmarks**: llama-bench used 14 threads (prefill) / 16 threads (decode) on AMD Ryzen 9 HX 370 and iMac M4; 6 threads on Galaxy A25 5G. Token rates reported with 500 prefill tokens and 250 output tokens.
>
> **GPU Benchmarks**: Metal (M2) and RTX 4090 numbers include both the backbone and ONNX codec decoder. The Metal M2 estimates are based on typical Metal performance vs CPU (2–4x speedup on memory-bound LLM inference).
>
> These benchmarks measure the speech language model and codec together, as used in a complete audio generation pipeline.

## Quick Start: Local UI

> [!TIP]
> **New in v1.2.0+**: One-command launcher for a local Gradio web UI. Perfect for trying models without writing code.

### Requirements
- Python 3.10–3.13
- On macOS: Xcode Command Line Tools (`xcode-select --install`)
- On Linux: espeak-ng (e.g., `sudo apt-get install espeak-ng`)

### Launch

```bash
# Clone the repo
git clone https://github.com/neuphonic/neutts.git
cd neutts

# Run the launcher (uses uv for environment management)
./run.sh

# Opens http://127.0.0.1:7860 in your browser
```

**First run**: If on Apple Silicon macOS, the script compiles `llama-cpp-python` with Metal support (~5–15 min, one-time only).

### Features

- **Model selector** — Choose from GGUF, PyTorch, or different codec backends
- **Device selection** — Auto-detect hardware (Metal/CUDA/CPU) or choose manually
- **Live validation** — Text length, reference audio duration checked in real-time
- **Streaming mode** — Watch audio generate chunk-by-chunk (GGUF models only)
- **Sample speakers** — Quick-start with built-in Dave, Greta, Jo, Juliette, or Mateo
- **Real-Time Factor (RTF) stats** — Monitor generation speed as it progresses
- **Sampling controls** — Tune temperature (0.1–2.0) and top-k (0–100) to trade quality for diversity

#### Advanced Usage

```bash
./run.sh --host 0.0.0.0 --port 8080        # Listen on all interfaces, port 8080
./run.sh --share                             # Create a public Gradio share link
./run.sh --host 192.168.1.100 --port 9000   # Custom host and port
```

To force a fresh Metal recompile (e.g., after updating llama-cpp-python):
```bash
rm .venv/.llama_metal && ./run.sh
```

## Get Started: Programmatic Usage

### 1. Install NeuTTS

```bash
pip install neutts
```

Or for local editable install:
```bash
git clone https://github.com/neuphonic/neutts.git
cd neutts
pip install -e .
```

To install all optional dependencies (GGUF + ONNX support):
```bash
pip install -e ".[all]"
```

### 2. (Optional) GPU Support

#### Apple Silicon (macOS M1–M4)

For GPU acceleration with Metal:
```bash
CMAKE_ARGS="-DGGML_METAL=ON" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

For CPU-only with Apple's Accelerate framework:
```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

#### NVIDIA CUDA

```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

#### AMD ROCm

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=ON" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

#### Linux CPU (OpenBLAS)

*Prerequisite: Install OpenBLAS (e.g., `sudo apt-get install libopenblas-dev`)*

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

#### Windows CPU (OpenBLAS)

*Prerequisite: Install OpenBLAS from [OpenBLAS releases](https://github.com/OpenMathLib/OpenBLAS/releases)*

```pwsh
$env:CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

### 3. (Optional) ONNX Codec Support

For minimal-latency inference with pre-encoded reference audio:
```bash
pip install "neutts[onnx]"
```

### One-Code Block Usage

```python
from neutts import NeuTTS
import soundfile as sf

# Initialize with auto-detected device
# Device options: "auto", "cpu", "cuda", "metal", "mps"
tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano-q8-gguf",  # GGUF for streaming; PyTorch for non-streaming
    backbone_device="auto",                          # Auto-detects Metal on M-series Macs
    codec_repo="neuphonic/neucodec-onnx-decoder",    # ONNX for latency; NeuCodec for quality
    codec_device="cpu",                              # ONNX runs on CPU only
)

# Prepare input
input_text = "My name is Andy. I'm 25 and I just moved to London."
ref_text = open("samples/jo.txt").read().strip()
ref_codes = tts.encode_reference("samples/jo.wav")

# Generate speech (with optional sampling control)
wav = tts.infer(
    input_text, 
    ref_codes, 
    ref_text,
    temperature=1.0,   # 0.1 = deterministic, 2.0 = creative
    top_k=50,          # 0 = disabled, limits sampling to top K tokens
)
sf.write("output.wav", wav, 24000)
```

### Streaming Example

```python
from neutts import NeuTTS
import soundfile as sf
import numpy as np

tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano-q8-gguf",
    backbone_device="auto",
    codec_repo="neuphonic/neucodec-onnx-decoder",
)

ref_codes = tts.encode_reference("samples/jo.wav")
ref_text = open("samples/jo.txt").read().strip()

# Streaming: audio generated in chunks, watch RTF in real-time
audio_chunks = []
for chunk in tts.infer_stream(
    "This is a long piece of text that will be streamed.",
    ref_codes,
    ref_text,
    temperature=1.0,
    top_k=50,
):
    audio_chunks.append(chunk)
    print(f"Generated {len(chunk) / 24000:.2f}s of audio so far...")

# Combine all chunks
full_audio = np.concatenate(audio_chunks)
sf.write("output.wav", full_audio, 24000)
```

> [!NOTE]
> Streaming is only available for **GGUF backbones** with `llama-cpp-python`. PyTorch models use non-streaming inference.

### Command-Line Examples

See `examples/` directory for more detailed scripts:

```bash
# Basic example
python -m examples.basic_example \
  --input_text "Hello world!" \
  --ref_audio samples/jo.wav \
  --ref_text samples/jo.txt \
  --backbone neuphonic/neutts-nano

# Streaming example (GGUF only)
python -m examples.basic_streaming_example \
  --input_text "This will be streamed!" \
  --ref_codes samples/jo.pt \
  --ref_text samples/jo.txt \
  --backbone neuphonic/neutts-nano-q8-gguf

# Minimal latency (ONNX decoder)
python -m examples.onnx_example \
  --input_text "Low latency!" \
  --ref_codes samples/jo.pt \
  --ref_text samples/jo.txt
```

## API Reference

### Device Strings

`NeuTTS` accepts the following device names for `backbone_device` and `codec_device`:

| String | Resolves To | Notes |
|--------|-------------|-------|
| `"auto"` | CUDA > Metal > CPU | Auto-detects best available hardware |
| `"cpu"` | CPU | Always available |
| `"cuda"` | NVIDIA CUDA | Requires CUDA-compiled llama-cpp-python |
| `"metal"` | Apple Metal (MPS) | Requires Metal-compiled llama-cpp-python; M-series Macs only |
| `"mps"` | Apple Metal (MPS) | Direct PyTorch Metal backend; same as `"metal"` |
| `"gpu"` | CUDA > Metal > CPU | Legacy alias; same fallback logic as `"auto"` |

**Note**: ONNX codec (`neucodec-onnx-decoder`, `neucodec-onnx-decoder-int8`) runs on CPU only. Full codecs (NeuCodec, DistillNeuCodec) respect the codec device.

### Sampling Parameters

`infer()` and `infer_stream()` now accept sampling controls:

```python
wav = tts.infer(
    text="...",
    ref_codes=codes,
    ref_text="...",
    temperature=1.0,  # 0.1 = deterministic, 2.0 = more random (default: 1.0)
    top_k=50,         # 0 = disabled, limit to top K tokens (default: 50)
)
```

- **`temperature`** (float, 0.1–2.0): Controls output diversity
  - 0.1 = highly deterministic (same input → same output)
  - 1.0 = balanced (default)
  - 2.0 = very random / creative
  
- **`top_k`** (int, 0–100): Limits sampling to the K most likely next tokens
  - 0 = disabled (sample from full distribution)
  - 50 = default (sample from top 50 tokens)
  - Higher = more diverse; lower = more focused

### Model Performance Notes

**Float16 on GPU**: PyTorch backbones automatically use float16 (half-precision) on CUDA and Metal/MPS, reducing memory by ~50% and improving speed ~2x compared to float32. No API changes required; it's automatic.

**Metal Flash Attention**: The GGUF backend disables flash attention on Metal (it only works on CUDA). This has negligible impact on latency for small models like NeuTTS-Nano.

**Max New Tokens**: Both `infer()` and `infer_stream()` use `max_new_tokens=2048`, ensuring the full context budget is available for generation regardless of prompt length. (Previous versions used `max_length`, which silently truncated output for long prompts.)

## Preparing References for Cloning

NeuTTS requires two inputs for voice cloning:

1. **Reference audio** (`.wav` file): A sample of the target voice
2. **Reference transcript**: Exact text of what is spoken in the audio

The model learns the voice characteristics from the audio and generates new speech in that style.

### Example Reference Files

Built-in samples in `samples/` directory:

**English**:
- `dave.wav` / `dave.txt` / `dave.pt`
- `jo.wav` / `jo.txt` / `jo.pt`

**Spanish**:
- `mateo.wav` / `mateo.txt` / `mateo.pt`

**German**:
- `greta.wav` / `greta.txt` / `greta.pt`

**French**:
- `juliette.wav` / `juliette.txt` / `juliette.pt`

### Guidelines for Best Results

For optimal voice cloning, reference audio should be:

1. **Mono channel** — Convert stereo to mono if needed
2. **16–44 kHz sample rate** — Most WAV files work; librosa handles resampling
3. **3–20 seconds in length** — At least 3 seconds; more captures richer characteristics
4. **Saved as WAV** — `.wav` format recommended
5. **Clean audio** — Minimal background noise; clear speech
6. **Natural, continuous speech** — Like a monologue or conversation with few pauses, so the model captures tone and pacing

### Pre-Encoding References

For latency-sensitive applications, pre-encode references once and reuse:

```python
import torch
from neutts import NeuTTS

tts = NeuTTS()

# Encode once
ref_codes = tts.encode_reference("samples/jo.wav")
torch.save(ref_codes, "jo_encoded.pt")

# Reuse many times
ref_codes = torch.load("jo_encoded.pt")
wav = tts.infer("Text 1", ref_codes, "Jo says...")
wav = tts.infer("Text 2", ref_codes, "Jo says...")  # No re-encoding
```

The UI (`app.py`) auto-caches encoded references by file path, so repeated use of the same speaker is instant.

## Guidelines for Minimizing Latency

For production on-device deployments:

1. **Use GGUF models** — Quantized models are faster and smaller than PyTorch
2. **Pre-encode references** — Save `.pt` files ahead of time; avoids encoder overhead
3. **Use ONNX codec** — Codec decoder accounts for ~20% of inference time; ONNX is optimized for speed
4. **Use Metal on Apple Silicon** — ~2–4x faster than CPU
5. **Use GPU when available** — CUDA (20+ tokens/s) vs CPU (100–200 tokens/s)

Example minimum-latency setup:

```python
tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano-q8-gguf",    # GGUF + quantized
    backbone_device="metal",                           # GPU if available
    codec_repo="neuphonic/neucodec-onnx-decoder",      # ONNX (CPU-only, minimal latency)
    codec_device="cpu",
)

# Pre-encode reference once
ref_codes = torch.load("reference_encoded.pt")

# Fast inference
wav = tts.infer("Text", ref_codes, "Reference text", temperature=1.0, top_k=50)
```

See `examples/onnx_example.py` for a complete latency-optimized example.

## Recent Changes (v1.2.0+)

### New Features

- **Apple Silicon Metal support** — Full GPU acceleration on M1–M4 Macs with `run.sh` launcher
- **Local Gradio UI** (`app.py`) — Web interface with model management, streaming visualization, sample speakers, and RTF stats
- **Sampling parameters** — Expose `temperature` and `top_k` on `infer()` and `infer_stream()`
- **Auto-device detection** — `"auto"` device string picks best available hardware at runtime
- **uv package manager integration** — `run.sh` uses uv for reproducible environments and easy Metal compilation

### Performance Optimizations

- **Cached special token IDs** — Eliminates tokenizer lookups on every inference call
- **Compiled regex** — Regex patterns compiled once at module load
- **Integer cache in streaming** — Reduces string operations per chunk; eliminates join + regex overhead
- **Watermark optimization** — Applied only to yielded audio, not overlapping decode windows; reduces watermarker calls
- **Max new tokens fix** — Changed from `max_length` to `max_new_tokens`; prevents silent truncation with long prompts
- **float16 on GPU** — Automatic half-precision for CUDA and Metal; ~2x speedup and 50% memory savings

### API Updates

- `infer()` signature: added `temperature` and `top_k` parameters
- `infer_stream()` signature: added `temperature` and `top_k` parameters
- Device strings: `"auto"`, `"metal"` (alias for `"mps"`), `"mps"` now supported
- PyTorch backbone: automatically uses float16 on GPU; no API change needed

### Bug Fixes

- **Metal flash attention** — Disabled on Metal (CUDA-only); was causing crashes
- **Device mismatch in `encode_reference`** — Audio tensor now moved to codec device
- **ONNX codec device handling** — Improved error messages; graceful fallback encoder loading
- **Long prompts truncation** — `max_new_tokens` ensures full context available regardless of prompt length

## Responsibility

Every audio file generated by NeuTTS includes by default a [Perth (Perceptual Threshold) Watermark](https://github.com/resemble-ai/perth) for authenticity verification.

> [!NOTE]
> If installing with `uv sync` in the repo and Perth watermarking fails (warning), install the package via PyPI instead (`pip install neutts`) to ensure watermarking is active.

## Disclaimer

Don't use this model to do bad things... please.

## Developer Setup

### Pre-Commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

For comprehensive testing (slow tests, downloads models):
```bash
RUN_SLOW=true pytest tests/
```

## Contributing

Contributions are welcome! Please ensure tests pass and pre-commit hooks are enabled before submitting pull requests.
