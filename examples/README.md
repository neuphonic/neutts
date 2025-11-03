# Examples

### GGUF Backbones

To run the model with `llama-cpp-python` in GGUF format, select a GGUF backbone when initialising the example script. All examples default to automatic device selection: CUDA (or DirectML/MPS where available) is chosen when possible, otherwise the CPU fallback is used.

```bash
python -m examples.basic_example \
<<<<<<< HEAD
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_audio ./samples/jo.wav \
  --ref_text ./samples/jo.txt \
  --backbone neuphonic/neutts-nano-q4-gguf
=======
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio ./samples/dave.wav \
  --ref_text ./samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf \
  --backbone_device auto \
  --codec_device auto
>>>>>>> ad81e78 (Enhance benchmarking capabilities and device selection in examples)
```

### Provider Benchmark

Benchmark the available ONNX Runtime execution providers and summarise their performance:

```bash
python -m examples.provider_benchmark \
  --input_text "Benchmarking NeuTTS Air" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --runs 3 \
  --warmup_runs 1 \
  --output benchmark_results.json \
  --summary_output benchmark_summary.txt
```

Add `--verbose` for per-run stats. `--summary_output` captures the rendered table and hardware information.

Set `--backbone_devices cpu,cuda` to measure both placements of a single backbone. Combine that with `--backbone_repos` to sweep multiple checkpoints in one pass. For example, the command below compares the torch checkpoint with the Q4 and Q8 GGUF exports across CPU/CUDA combinations for both the backbone and codec:

```bash
python -m examples.provider_benchmark \
  --input_text "Benchmarking NeuTTS Air combos" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone_repos neuphonic/neutts-air,neuphonic/neutts-air-q4-gguf,neuphonic/neutts-air-q8-gguf \
  --backbone_devices cpu,cuda \
  --codec_devices cpu,cuda \
  --runs 3 \
  --warmup_runs 1 \
  --output gguf_comparison.json \
  --summary_output gguf_comparison.txt
```

Use `--codec_repos` if you need to include additional decoder builds alongside `neucodec-onnx-decoder`. Each device is instantiated once and a warm-up pass (configurable via `--warmup_runs`) is executed before measurements start. Add `--no_reuse` to revert to the legacy behaviour that reloads models for every run.

### Pre-encode a reference

Reference encoding can be done ahead of time to reduce latency whilst inferencing the model; to pre-encode a reference you only need to provide a reference audio, as in the following script:

```bash
python -m examples.encode_reference \
 --ref_audio  ./samples/jo.wav \
 --output_path ./samples/jo.pt
 ```

Note that `basic_streaming_example.py` requires a pre-encoded reference. `basic_example.py` will encode your reference if a pre-encoding does not exist, and will save and use it in future runs with the same reference.

### Minimal Latency Example

To take advantage of encoding references ahead of time, we have a compiled the codec decoder into an [onnx graph](https://huggingface.co/neuphonic/neucodec-onnx-decoder) that enables inferencing NeuTTS without loading the encoder.
This can be useful for running the model in resource-constrained environments where the encoder may add a large amount of extra latency/memory usage.

To test the decoder, install a matching ONNX Runtime build for your platform (examples below) and run one of the following:

```bash
python -m examples.basic_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_audio samples/jo.wav \
  --ref_text samples/jo.txt \
  --backbone neuphonic/neutts-nano-q4-gguf \
  --codec neuphonic/neucodec-onnx-decoder
```

<<<<<<< HEAD
Since the onnx codec is decoder-only, this requires the pre-encoded reference (e.g. `samples/jo.pt`) to already exist alongside the wav.
=======
```bash
python -m examples.onnx_example_gpu \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf \
  --codec_device cuda
```

Set `--codec_device` to `auto` (default) to pick the first available GPU provider with an automatic CPU fallback. Use `cuda`/`cuda:0` (or `directml`, `rocm`) to request a specific provider—each will still fall back to CPU if unavailable. Pass `cpu` to disable GPU execution entirely.

Mac / Apple Silicon notes
-------------------------

On Apple M1/M2/M3 machines there isn't a package named `onnxruntime-gpu` like on Linux/Windows. Recent official `onnxruntime` wheels for macOS may include native Metal/CoreML providers, and there are community builds (for example `onnxruntime-silicon` / `onnxruntime-metal`) that target Apple Silicon specifically.

Suggested install steps for macOS (try them in order):

```bash
# 1) Try the official wheel first
pip install --upgrade pip
pip install onnxruntime

# 2) If the official wheel doesn't expose Metal/CoreML providers, try a community wheel
# (see https://github.com/cansik/onnxruntime-silicon for details). Example (may require a specific wheel URL):
# pip install onnxruntime-silicon

# 3) As a last resort, build ONNX Runtime from source with Metal/CoreML enabled following
# the official ONNX Runtime build docs: https://onnxruntime.ai/docs/build/.
```

Verify what providers are available with this tiny check:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

If the output contains `MetalExecutionProvider` or `CoreMLExecutionProvider`, you can request those via `--codec_device metal` or `--codec_device coreml` respectively. If no GPU providers are present the code will correctly fall back to `CPUExecutionProvider`.

### Streaming Support 
>>>>>>> 98ebc11 (Add ONNX decoder GPU support with CPU fallback)

### Streaming Support

To stream the model output in chunks, try out the `basic_streaming_example.py` example. For streaming, only the GGUF backbones are currently supported. Ensure you have `llama-cpp-python`, `onnxruntime` and `pyaudio` installed to run this example. `pyaudio` needs the PortAudio library on macOS and Linux:

```bash
# macOS
brew install portaudio && pip install pyaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev && pip install pyaudio

# Windows (prebuilt wheels)
pip install pyaudio
```

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_codes samples/jo.pt \
  --ref_text samples/jo.txt \
  --backbone neuphonic/neutts-nano-q4-gguf
```

### Emotional TTS (NeuTTS-2E)

NeuTTS-2E uses fixed speakers with bundled pre-encoded references, so the emotional examples take a `--speaker` (`emily`, `paul`, `sophie`, `steven`) and an `--emotion` (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`) instead of reference arguments:

```bash
python -m examples.basic_example_emotions \
  --input_text "I can't believe it's finally here!" \
  --speaker emily \
  --emotion happy
```

Streaming (GGUF backbones, `neuphonic/neutts-2e-q8-gguf` by default):

```bash
python -m examples.basic_streaming_example_emotions \
  --input_text "I can't believe it's finally here!" \
  --speaker emily \
  --emotion happy \
  --backbone neuphonic/neutts-2e-q4-gguf
```
