# Copilot Instructions for `neutts-air`

## Context Snapshot
- Ignore the repo-root `instructions.md` (it references an unrelated VibeType project); use this document and the in-repo docs instead.
- `neuttsair/neutts.py` holds the primary `NeuTTSAir` implementation (tokenisation, backbone loading, codec orchestration, streaming, watermarking).
- `neuttsair/benchmark.py` provides the benchmarking CLI and utility helpers reused by `examples/provider_benchmark.py`.
- `examples/` contains runnable entry points (`basic_example.py`, `basic_streaming_example.py`, ONNX demos) that showcase supported workflows.
- `tests/` currently focuses on device negotiation (`test_device_selection.py`) and cached reference encoding (`test_reference_cache.py`).

## Key Design Points
- Backbones can be Hugging Face transformers or GGUF/llama.cpp exports; codec backends can be Torch (`NeuCodec`, `DistillNeuCodec`) or ONNX (`NeuCodecOnnxDecoder`). Code often checks `.encode_code` presence to decide which path to use.
- Device selection defaults to "auto". Helper methods `_select_torch_device`, `_select_backbone_device`, and `_configure_onnx_codec_session` encapsulate GPU/MPS/DirectML fallbacks, and emit warnings when the user requests unsupported providers.
- `encode_reference` now supports caching: audio is encoded once, cached in-memory, and optionally persisted via `codes_path`. When the ONNX decoder is active, a lazy Torch encoder is spun up for the encode step only.
- Watermarking is mandatory: inference paths call `PerthImplicitWatermarker.apply_watermark` before returning audio.
- Streaming currently exists only for GGUF/llama.cpp backbones; Torch streaming raises `NotImplementedError`.

## Conventions & Patterns
- Prefer path-like arguments (`Path`, strings) that resolve/expanduser before IO. Recent changes cache using `_ReferenceCacheKey(path, mtime_ns, size)` to invalidate stale files.
- Keep new helpers private (`_foo`) when they relate to internals (device selection, caching). Public API surface is intentionally small (`NeuTTSAir`, benchmarks module, examples CLI entrypoints).
- Unit tests mock external libraries (`torch.cuda`, `librosa`, `onnxruntime`) aggressively—use `patch` and temporary directories to keep tests hermetic.

## Common Workflows
- Run PyTorch inference example: `python -m examples.basic_example --input_text "Hello" --ref_audio samples/dave.wav --ref_text samples/dave.txt --ref_codes cache/dave.pt` (auto-creates cache on first run).
- Stream with GGUF + ONNX decoder: `python -m examples.basic_streaming_example --backbone neuphonic/neutts-air-q8-gguf --ref_codes samples/dave.pt`.
- Benchmark providers: `python -m examples.provider_benchmark --input_text "Benchmark" --ref_codes samples/dave.pt --runs 3 --warmup_runs 1 --summary_output artifacts/benchmarks/run.txt`.
- Unit tests: `pytest tests/test_device_selection.py tests/test_reference_cache.py` (full suite is fast, ~10 tests).

## Integration & Dependencies
- Install base deps via `pip install -r requirements.txt`. GPU/ONNX workflows may need `onnxruntime-gpu` or `onnxruntime-directml` plus `pip install -r requirements-gpu.txt`.
- GGUF paths require `llama-cpp-python`; streaming example additionally needs `pyaudio`.
- The codec encoder (`NeuCodec`) is only available via Torch; ONNX builds ship decoder-only graphs. Leverage `_ensure_reference_encoder` when ONNX is active.

## Pitfalls / Tips
- `.gitignore` excludes `tests/` by default; stage new tests with `git add -f tests/...` or adjust the ignore rules locally.
- Always pass `codes_path` when writing demos or benchmarks that loop, otherwise reference encoding will dominate latency.
- When extending device logic, update both implementation and coverage in `tests/test_device_selection.py` to keep regressions from sneaking in.
- Ensure warnings stay informative—user workflows rely on them when GPU providers are missing.

## User Requests: 
use the speak MCP to speak often , be chatty talk about what you doing and why as you do it. 


_Questions or gaps? Ping the maintainers to refine these instructions._
