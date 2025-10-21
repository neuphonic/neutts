# Changelog

All notable changes in this fork are documented here to help prepare an upstream pull request.

## [Unreleased] - 2025-10-15

### Added
- Automatic device selection helpers in `neuttsair/neutts.py` that prefer CUDA/MPS for the backbone and GPU execution providers for the ONNX codec while gracefully falling back to CPU.
- Regression tests in `tests/test_device_selection.py` covering torch device selection and ONNX provider configuration.
- Benchmark utilities in `neuttsair/benchmark.py` plus the `examples/provider_benchmark.py` CLI for profiling provider combinations, along with unit tests in `tests/test_benchmark.py`.
- A GPU-oriented requirements lock-in (`requirements-gpu.txt`) that captures the dependency set used to validate CUDA ONNX Runtime benchmarks.

### Changed
- Updated ONNX codec configuration to accept `auto`, `cuda`, `directml`, `rocm`, etc., and ensure CPU fallback when providers are unavailable.
- Simplified example scripts (`examples/basic_example.py`, `examples/onnx_example.py`, `examples/basic_streaming_example.py`, `examples/onnx_example_gpu.py`) to rely on auto device selection.
- Refreshed `README.md` and `examples/README.md` with installation notes and explanations of the new device behaviour.
- Expanded documentation to cover multi-backbone benchmarking (Torch, Q4 GGUF, Q8 GGUF) and the new CLI options (`--backbone_repos`, `--codec_repos`, extended device lists).
- Documented benchmarking workflow in the README files.
- Benchmarks now reuse instantiated models per device with configurable warm-up passes (`--warmup_runs`) and an opt-out flag (`--no_reuse`), yielding more stable measurements while avoiding redundant downloads.
- Provider benchmarking CLI can now sweep multiple backbone placements via `--backbone_devices`, presenting results with separate backbone/codec columns and serialising combination metadata to JSON outputs.
- Consolidated the benchmarking CLI logic inside `neuttsair/benchmark.py` while keeping `examples/provider_benchmark.py` as a thin wrapper.
- Repository housekeeping adds a dedicated `artifacts/` tree (with `.gitkeep` placeholders) and an expanded `.gitignore` so generated audio/benchmark outputs stay out of version control by default.

### Notes
- Store generated audio and benchmark artifacts under `artifacts/` (these remain ignored by default) and omit them from commits.
