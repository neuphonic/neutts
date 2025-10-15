# Changelog

All notable changes in this fork are documented here to help prepare an upstream pull request.

## [Unreleased] - 2025-10-13

### Added
- Automatic device selection helpers in `neuttsair/neutts.py` that prefer CUDA/MPS for the backbone and GPU execution providers for the ONNX codec while gracefully falling back to CPU.
- Regression tests in `tests/test_device_selection.py` covering torch device selection and ONNX provider configuration.
- Benchmark utilities in `neuttsair/benchmark.py` plus the `examples/provider_benchmark.py` CLI for profiling provider combinations, along with unit tests in `tests/test_benchmark.py`.

### Changed
- Updated ONNX codec configuration to accept `auto`, `cuda`, `directml`, `rocm`, etc., and ensure CPU fallback when providers are unavailable.
- Simplified example scripts (`examples/basic_example.py`, `examples/onnx_example.py`, `examples/basic_streaming_example.py`, `examples/onnx_example_gpu.py`) to rely on auto device selection.
- Refreshed `README.md` and `examples/README.md` with installation notes and explanations of the new device behaviour.
- Expanded documentation to cover multi-backbone benchmarking (Torch, Q4 GGUF, Q8 GGUF) and the new CLI options (`--backbone_repos`, `--codec_repos`, extended device lists).
- Documented benchmarking workflow in the README files.
- Benchmarks now reuse instantiated models per device with configurable warm-up passes (`--warmup_runs`) and an opt-out flag (`--no_reuse`), yielding more stable measurements while avoiding redundant downloads.
- Provider benchmarking CLI can now sweep multiple backbone placements via `--backbone_devices`, presenting results with separate backbone/codec columns and serialising combination metadata to JSON outputs.

### Notes
- Generated audio artifacts (`output.wav`, `output_onnx.wav`, `output_onnx_gpu.wav`) are local test outputs and should be excluded from commits.
