# Pull Request Draft: ONNX GPU Support

## Summary
- Extend `NeuTTSAir` to auto-select CUDA/MPS/CPU for the backbone and ONNX codec, configuring CUDA, DirectML, or ROCm providers when present and falling back to CPU with clear warnings when unavailable.
- Document the new GPU workflow in `README.md`, `examples/README.md`, and the freshly added `examples/onnx_example_gpu.py`; ship `requirements-gpu.txt` for quick GPU setup.
- Publish the new benchmarking suite (CLI plus artifacts) and introduce `tests/test_device_selection.py` to cover device routing and ONNX provider selection so regressions surface quickly.

## Benchmarks
*Windows 11 · NVIDIA GeForce GTX 1080 Ti*

| Backbone Repo | Backbone Device | Codec Repo | Codec Device | Providers | Runs | Load (s) | Infer (s) | Total (s) | RTF | RAM (MB) | VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neuphonic/neutts-air | cpu | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 15.80 ± 22.34 | 44.23 ± 3.63 | 60.02 ± 23.28 | 12.31 ± 0.97 | 1,024 ± 1,437 | 0 |
| neuphonic/neutts-air | cpu | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 19.57 ± 27.68 | 41.51 ± 2.25 | 61.08 ± 28.12 | 12.45 ± 0.04 | 317 ± 444 | 0 |
| neuphonic/neutts-air | cuda | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 24.40 ± 34.51 | 7.89 ± 0.51 | 32.30 ± 34.00 | 2.87 ± 0.02 | 1.3 ± 0.6 | 2,890 ± 994 |
| neuphonic/neutts-air | cuda | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 31.05 ± 43.91 | 7.92 ± 0.94 | 38.97 ± 43.50 | 2.48 ± 0.02 | 47 ± 64 | 2,891 ± 995 |
| neuphonic/neutts-air-q4-gguf | cpu | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 1.61 ± 2.28 | 3.60 ± 0.20 | 5.20 ± 2.47 | 1.44 ± 0.07 | 394 ± 556 | 8.1 |
| neuphonic/neutts-air-q4-gguf | cpu | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 1.28 ± 1.82 | 3.21 ± 0.02 | 4.50 ± 1.83 | 1.29 ± 0.01 | 151 ± 213 | 8.1 |
| neuphonic/neutts-air-q4-gguf | cuda | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 1.26 ± 1.79 | 3.74 ± 0.59 | 5.01 ± 2.23 | 1.40 ± 0.03 | 393 ± 555 | 8.1 |
| neuphonic/neutts-air-q4-gguf | cuda | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 1.27 ± 1.79 | 3.63 ± 0.50 | 4.90 ± 2.17 | 1.37 ± 0.06 | 152 ± 214 | 8.1 |
| neuphonic/neutts-air-q8-gguf | cpu | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 1.41 ± 1.99 | 7.15 ± 1.66 | 8.55 ± 2.28 | 1.81 ± 0.08 | 273 ± 373 | 8.1 |
| neuphonic/neutts-air-q8-gguf | cpu | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 1.37 ± 1.93 | 6.65 ± 1.40 | 8.02 ± 1.96 | 1.69 ± 0.03 | 30 ± 40 | 8.1 |
| neuphonic/neutts-air-q8-gguf | cuda | neuphonic/neucodec-onnx-decoder | cpu | CPUExecutionProvider | 3 | 1.33 ± 1.87 | 5.12 ± 0.74 | 6.44 ± 2.60 | 1.75 ± 0.02 | 268 ± 378 | 8.1 |
| neuphonic/neutts-air-q8-gguf | cuda | neuphonic/neucodec-onnx-decoder | cuda | CUDAExecutionProvider, CPUExecutionProvider | 3 | 1.29 ± 1.83 | 5.00 ± 0.90 | 6.30 ± 2.72 | 1.70 ± 0.04 | 40 ± 56 | 8.1 |

## Testing
- `pytest tests/test_device_selection.py`

## PR response note

You can paste this into the PR reply to make the device-selection behaviour explicit for reviewers:

"Note: The device‑selection logic in `neuttsair/benchmark.py` is the same logic used in `neuttsair/neutts.py` for all inference. In short: if a GPU provider like CUDA, ROCm, DirectML, Metal or CoreML is available it will be used; otherwise the system falls back to CPU. On Apple Silicon (M1/M2/M3) without a Metal/CoreML ONNX runtime the system will run on CPU — this applies not only to the benchmark but to the whole model pipeline."


