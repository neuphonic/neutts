from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Optional, Sequence
import gc
import time

import numpy as np
import psutil  # type: ignore[import-not-found]
import torch

from .neutts import NeuTTSAir

_MB = 1024 * 1024


@dataclass(slots=True)
class BenchmarkSample:
    """Container describing the data required for a benchmark run."""

    input_text: str
    ref_codes: Any
    ref_text: str


@dataclass(slots=True)
class BenchmarkResult:
    """Captured metrics for a single codec-device benchmark run."""

    codec_device: str
    providers: Optional[list[str]]
    load_s: float
    inference_s: float
    total_s: float
    audio_seconds: float
    realtime_factor: Optional[float]
    ram_mb: float
    vram_mb: Optional[float]


def candidate_codec_devices(
    explicit: Sequence[str] | None = None,
    *,
    ort_module: Any | None = None,
) -> list[str]:
    """Return a deduplicated list of codec device strings worth benchmarking.

    If ``explicit`` is provided, its values (in order) are returned. Otherwise,
    the list is inferred from the available ONNX Runtime execution providers. A
    CPU-only fallback is always included.
    """

    if explicit:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in explicit:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    if ort_module is None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return ["cpu"]
    else:
        ort = ort_module

    try:
        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - defensive guard
        return ["cpu"]

    devices: list[str] = []
    if available:
        devices.append("auto")

    if "CUDAExecutionProvider" in available:
        devices.append("cuda")
        try:
            device_count = torch.cuda.device_count()
            for idx in range(device_count):
                devices.append(f"cuda:{idx}")
        except Exception:  # pragma: no cover - fallback for CPU-only torch builds
            pass

    if "ROCMExecutionProvider" in available:
        devices.append("rocm")

    if "DmlExecutionProvider" in available:
        devices.append("directml")

    # Apple Silicon / macOS ONNX providers
    if "MetalExecutionProvider" in available:
        devices.append("metal")
    if "CoreMLExecutionProvider" in available:
        devices.append("coreml")

    devices.append("cpu")

    # Deduplicate while preserving order
    ordered: list[str] = []
    seen: set[str] = set()
    for item in devices:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _prepare_cuda(codec_device: str) -> Optional[int]:
    if not torch.cuda.is_available():
        return None

    device_index: Optional[int]
    if codec_device.startswith("cuda:"):
        try:
            device_index = int(codec_device.split(":", 1)[1])
        except ValueError:
            device_index = 0
    else:
        if torch.cuda.is_initialized():
            device_index = torch.cuda.current_device()
        else:
            device_index = 0

    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    return device_index


def _run_inference(
    tts: Any,
    sample: BenchmarkSample,
    *,
    process: psutil.Process,
    cuda_device_index: Optional[int],
) -> tuple[float, float, Optional[float], float, Optional[float]]:
    """Execute a single inference pass and return timing/memory metrics."""

    if cuda_device_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device_index)

    ram_before = process.memory_info().rss
    infer_start = time.perf_counter()
    wav = tts.infer(sample.input_text, sample.ref_codes, sample.ref_text)
    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)
    inference_s = time.perf_counter() - infer_start

    audio_seconds = 0.0
    if wav is not None:
        try:
            audio_seconds = float(len(wav)) / float(tts.sample_rate)
        except Exception:  # pragma: no cover - safeguard for unexpected shapes
            audio_seconds = 0.0

    realtime_factor: Optional[float]
    if audio_seconds > 0:
        realtime_factor = inference_s / audio_seconds
    else:
        realtime_factor = None

    ram_after = process.memory_info().rss
    ram_mb = max(0.0, (ram_after - ram_before) / _MB)

    vram_mb: Optional[float] = None
    if cuda_device_index is not None:
        try:
            peak_bytes = torch.cuda.max_memory_allocated(cuda_device_index)
            vram_mb = peak_bytes / _MB
        except Exception:  # pragma: no cover - CUDA edge cases
            vram_mb = None

    return inference_s, audio_seconds, realtime_factor, ram_mb, vram_mb


def _benchmark_device_runs(
    codec_device: str,
    *,
    sample: BenchmarkSample,
    runs: int,
    warmup_runs: int,
    backbone_repo: str,
    backbone_device: str,
    codec_repo: str,
    tts_factory: Callable[..., Any],
    process: psutil.Process | None,
) -> list[BenchmarkResult]:
    """Benchmark ``codec_device`` by reusing a single model instance."""

    process = process or psutil.Process()

    cuda_device_index = _prepare_cuda(codec_device)

    gc.collect()
    ram_before_load = process.memory_info().rss
    if cuda_device_index is not None:
        base_vram_bytes = torch.cuda.memory_allocated(cuda_device_index)
    else:
        base_vram_bytes = 0

    load_start = time.perf_counter()
    tts = tts_factory(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )
    load_s = time.perf_counter() - load_start

    ram_after_load = process.memory_info().rss
    load_ram_mb = max(0.0, (ram_after_load - ram_before_load) / _MB)

    if cuda_device_index is not None:
        current_vram_bytes = torch.cuda.memory_allocated(cuda_device_index)
        load_vram_mb = max(0.0, (current_vram_bytes - base_vram_bytes) / _MB)
    else:
        load_vram_mb = 0.0

    providers: Optional[list[str]] = None
    session = getattr(getattr(tts, "codec", None), "session", None)
    if session and hasattr(session, "get_providers"):
        try:
            providers = list(session.get_providers())
        except Exception:  # pragma: no cover - defensive guard
            providers = None

    warmup_runs = max(0, warmup_runs)
    for _ in range(warmup_runs):
        _run_inference(
            tts,
            sample,
            process=process,
            cuda_device_index=cuda_device_index,
        )

    measured_runs = max(1, runs)
    results: list[BenchmarkResult] = []
    for run_index in range(measured_runs):
        inference_s, audio_seconds, realtime_factor, ram_mb, vram_mb = _run_inference(
            tts,
            sample,
            process=process,
            cuda_device_index=cuda_device_index,
        )

        if run_index == 0:
            ram_mb += load_ram_mb
            if vram_mb is not None:
                vram_mb += load_vram_mb

        total_s = inference_s + (load_s if run_index == 0 else 0.0)
        results.append(
            BenchmarkResult(
                codec_device=codec_device,
                providers=providers,
                load_s=load_s if run_index == 0 else 0.0,
                inference_s=inference_s,
                total_s=total_s,
                audio_seconds=audio_seconds,
                realtime_factor=realtime_factor,
                ram_mb=ram_mb,
                vram_mb=vram_mb,
            )
        )

    # Tidy up as much as possible before returning
    del tts
    gc.collect()
    if cuda_device_index is not None:
        torch.cuda.empty_cache()

    return results


def benchmark_codec_device(
    codec_device: str,
    *,
    sample: BenchmarkSample,
    backbone_repo: str = "neuphonic/neutts-air",
    backbone_device: str = "auto",
    codec_repo: str = "neuphonic/neucodec-onnx-decoder",
    tts_factory: Callable[..., Any] = NeuTTSAir,
    process: psutil.Process | None = None,
) -> BenchmarkResult:
    """Benchmark a single codec device configuration."""

    return _benchmark_device_runs(
        codec_device,
        sample=sample,
        runs=1,
        warmup_runs=0,
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        tts_factory=tts_factory,
        process=process,
    )[0]


def summarise_metrics(results: Sequence[BenchmarkResult]) -> dict[str, float]:
    """Return aggregate statistics (mean + population stddev) for the results."""

    if not results:
        return {}

    numeric_fields = [
        "load_s",
        "inference_s",
        "total_s",
        "audio_seconds",
        "realtime_factor",
        "ram_mb",
        "vram_mb",
    ]

    summary: dict[str, float] = {}
    for field in numeric_fields:
        values = [getattr(result, field) for result in results if getattr(result, field) is not None]
        if not values:
            continue
        summary[f"{field}_mean"] = float(mean(values))
        summary[f"{field}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return summary


def iter_benchmarks(
    codecs: Iterable[str],
    *,
    sample: BenchmarkSample,
    runs: int = 1,
    warmup_runs: int = 1,
    reuse_models: bool = True,
    **kwargs: Any,
) -> dict[str, list[BenchmarkResult]]:
    """Benchmark several codec devices and return raw results grouped by device."""

    grouped: dict[str, list[BenchmarkResult]] = {}

    for codec_device in codecs:
        if reuse_models:
            grouped[codec_device] = _benchmark_device_runs(
                codec_device,
                sample=sample,
                runs=max(1, runs),
                warmup_runs=max(0, warmup_runs),
                backbone_repo=kwargs.get("backbone_repo", "neuphonic/neutts-air"),
                backbone_device=kwargs.get("backbone_device", "auto"),
                codec_repo=kwargs.get("codec_repo", "neuphonic/neucodec-onnx-decoder"),
                tts_factory=kwargs.get("tts_factory", NeuTTSAir),
                process=kwargs.get("process"),
            )
        else:
            device_results: list[BenchmarkResult] = []
            for _ in range(max(1, runs)):
                device_results.append(benchmark_codec_device(codec_device, sample=sample, **kwargs))
            grouped[codec_device] = device_results
    return grouped


def _load_ref_codes(path: Path) -> Sequence[float] | np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Reference codes file not found: {path}")

    if path.suffix == ".pt":
        try:
            codes = torch.load(path, map_location="cpu")
        except Exception:
            codes = np.load(path, allow_pickle=True)
    elif path.suffix in {".npy", ".npz"}:
        codes = np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported reference code format: {path.suffix}")

    return codes


def _load_ref_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Reference transcript not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _format_metric(summary: dict[str, float], key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if mean_key not in summary:
        return "n/a"
    mean_val = summary.get(mean_key, float("nan"))
    std_val = summary.get(std_key, float("nan"))
    return f"{mean_val:.3f} ± {std_val:.3f}"


def _format_providers(providers: list[str] | None) -> str:
    if not providers:
        return "n/a"
    joined = ", ".join(providers)
    if len(joined) <= 48:
        return joined
    return "\n".join(f"- {provider}" for provider in providers)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    split_rows: list[list[list[str]]] = []
    for row in rows:
        split_row: list[list[str]] = []
        for idx, cell in enumerate(row):
            lines = cell.splitlines() or [""]
            split_row.append(lines)
            for line in lines:
                widths[idx] = max(widths[idx], len(line))
        split_rows.append(split_row)

    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))

    body_lines: list[str] = []
    for split_row in split_rows:
        max_lines = max(len(lines) for lines in split_row)
        for line_idx in range(max_lines):
            parts = []
            for col_idx, lines in enumerate(split_row):
                text = lines[line_idx] if line_idx < len(lines) else ""
                parts.append(text.ljust(widths[col_idx]))
            body_lines.append("  ".join(parts))

    return "\n".join([header_line, divider_line, *body_lines])


def _collect_system_metadata(
    *,
    platform_module=platform,
    psutil_module=psutil,
    torch_module=torch,
) -> dict[str, list[str] | str | None]:
    info: dict[str, list[str] | str | None] = {}

    info["os"] = platform_module.platform()
    processor = platform_module.processor() or platform_module.machine()
    info["cpu"] = processor
    logical = psutil_module.cpu_count(logical=True)
    physical = psutil_module.cpu_count(logical=False) or logical
    info["cpu_counts"] = f"{physical} physical / {logical} logical"

    try:
        total_ram = psutil_module.virtual_memory().total
        info["ram"] = f"{total_ram / (1024 ** 3):.1f} GB"
    except Exception:
        info["ram"] = None

    gpus: list[str] = []
    try:
        if torch_module.cuda.is_available():
            for idx in range(torch_module.cuda.device_count()):
                name = torch_module.cuda.get_device_name(idx)
                capability = torch_module.cuda.get_device_capability(idx)
                gpus.append(f"cuda:{idx} {name} (sm_{capability[0]}{capability[1]})")
    except Exception:
        gpus = []

    info["gpus"] = gpus
    return info


def _format_system_metadata(metadata: dict[str, list[str] | str | None]) -> list[str]:
    lines: list[str] = []
    lines.append(f"OS: {metadata.get('os', 'unknown')}")
    cpu = metadata.get("cpu") or "unknown"
    counts = metadata.get("cpu_counts")
    cpu_line = f"CPU: {cpu}"
    if counts:
        cpu_line += f" ({counts})"
    lines.append(cpu_line)
    ram = metadata.get("ram")
    if ram:
        lines.append(f"RAM: {ram}")
    gpus = metadata.get("gpus") or []
    if gpus:
        lines.append("GPUs:")
        for gpu in gpus:  # type: ignore[arg-type]
            lines.append(f"  - {gpu}")
    else:
        lines.append("GPUs: none detected")
    return lines


def _store_results(
    path: Path,
    grouped_results: dict[tuple[str, str, str, str], list[BenchmarkResult]],
    summaries: dict[tuple[str, str, str, str], dict[str, float]],
) -> None:
    serialisable: list[dict[str, object]] = []
    for (backbone_repo, backbone_device, codec_repo, codec_device), results in grouped_results.items():
        serialisable.append(
            {
                "backbone_repo": backbone_repo,
                "backbone_device": backbone_device,
                "codec_repo": codec_repo,
                "codec_device": codec_device,
                "providers": results[0].providers if results else None,
                "runs": [asdict(result) for result in results],
                "summary": summaries.get((backbone_repo, backbone_device, codec_repo, codec_device), {}),
            }
        )
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def _parse_list_option(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _deduplicate(sequence: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in sequence:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _resolve_backbone_devices(backbone_repo: str, override: list[str] | None) -> list[str]:
    if override:
        return _deduplicate(override)
    return ["auto"]


def _resolve_codec_devices(codec_repo: str, override: list[str] | None) -> list[str]:
    if override:
        return _deduplicate(override)

    if codec_repo == "neuphonic/neucodec-onnx-decoder":
        return candidate_codec_devices()

    devices: list[str] = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.insert(1, "cuda")
    return devices


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark NeuTTS Air backbones and codecs")
    parser.add_argument(
        "--input_text",
        type=str,
        default="Testing NeuTTS Air benchmarking",
        help="Text to synthesise during the benchmark",
    )
    parser.add_argument(
        "--ref_codes",
        type=Path,
        default=Path("samples/dave.pt"),
        help="Path to pre-encoded reference codes",
    )
    parser.add_argument(
        "--ref_text",
        type=Path,
        default=Path("samples/dave.txt"),
        help="Path to the reference transcript",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-air",
        help="Backbone repository to use for the benchmark",
    )
    parser.add_argument(
        "--backbone_repos",
        type=str,
        default=None,
        help="Comma-separated list of backbone repositories to benchmark (overrides --backbone)",
    )
    parser.add_argument(
        "--backbone_device",
        type=str,
        default="auto",
        help="Backbone device override (defaults to auto)",
    )
    parser.add_argument(
        "--backbone_devices",
        type=str,
        default=None,
        help="Comma-separated list of backbone devices to benchmark (overrides --backbone_device)",
    )
    parser.add_argument(
        "--codec_devices",
        type=str,
        default=None,
        help="Comma-separated list of codec devices to benchmark (defaults to auto detection)",
    )
    parser.add_argument(
        "--codec_repos",
        type=str,
        default=None,
        help="Comma-separated list of codec repositories to benchmark (defaults to the ONNX decoder)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per codec device (averaged in the summary)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warm-up passes (not measured) to run before timing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store raw benchmark data",
    )
    parser.add_argument(
        "--no_reuse",
        action="store_true",
        help="Disable model reuse between runs (falls back to legacy behaviour)",
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=None,
        help="Optional text file to store the rendered summary table",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run metrics instead of only the summary",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    ref_codes = _load_ref_codes(args.ref_codes)
    ref_text = _load_ref_text(args.ref_text)

    sample = BenchmarkSample(
        input_text=args.input_text,
        ref_codes=ref_codes,
        ref_text=ref_text,
    )

    backbone_repos = _parse_list_option(args.backbone_repos) or [args.backbone]
    codec_repos = _parse_list_option(args.codec_repos) or ["neuphonic/neucodec-onnx-decoder"]
    backbone_device_overrides = _parse_list_option(args.backbone_devices) or [args.backbone_device]
    codec_device_overrides = _parse_list_option(args.codec_devices)

    process = psutil.Process()

    grouped_results: dict[tuple[str, str, str, str], list[BenchmarkResult]] = {}

    for backbone_repo in backbone_repos:
        backbone_devices = _resolve_backbone_devices(backbone_repo, backbone_device_overrides)
        for backbone_device in backbone_devices:
            for codec_repo in codec_repos:
                codec_devices = _resolve_codec_devices(codec_repo, codec_device_overrides)
                if not codec_devices:
                    continue

                try:
                    partial_results = iter_benchmarks(
                        codec_devices,
                        sample=sample,
                        runs=max(1, args.runs),
                        warmup_runs=max(0, args.warmup_runs),
                        reuse_models=not args.no_reuse,
                        backbone_repo=backbone_repo,
                        backbone_device=backbone_device,
                        codec_repo=codec_repo,
                        process=process,
                    )
                except Exception as exc:
                    print(
                        f"Skipping combination {backbone_repo} ({backbone_device}) with {codec_repo}: {exc}"
                    )
                    continue

                for codec_device, results in partial_results.items():
                    if results:
                        grouped_results[(backbone_repo, backbone_device, codec_repo, codec_device)] = results

    summaries: dict[tuple[str, str, str, str], dict[str, float]] = {}

    metadata = _collect_system_metadata()
    metadata_lines = _format_system_metadata(metadata)

    print("\nSystem information:\n")
    for line in metadata_lines:
        print(line)

    print("\nBenchmark summary (mean ± standard deviation):\n")
    headers = [
        "Backbone Repo",
        "Backbone Device",
        "Codec Repo",
        "Codec Device",
        "Providers",
        "Runs",
        "Load (s)",
        "Infer (s)",
        "Total (s)",
        "RTF",
        "RAM (MB)",
        "VRAM (MB)",
    ]

    rows: list[list[str]] = []
    for backbone_repo, backbone_device, codec_repo, codec_device in sorted(grouped_results.keys()):
        results = grouped_results[(backbone_repo, backbone_device, codec_repo, codec_device)]
        if not results:
            continue

        summary = summarise_metrics(results)
        summaries[(backbone_repo, backbone_device, codec_repo, codec_device)] = summary
        providers = _format_providers(results[0].providers if results else None)

        rows.append(
            [
                backbone_repo,
                backbone_device,
                codec_repo,
                codec_device,
                providers,
                str(len(results)),
                _format_metric(summary, "load_s"),
                _format_metric(summary, "inference_s"),
                _format_metric(summary, "total_s"),
                _format_metric(summary, "realtime_factor"),
                _format_metric(summary, "ram_mb"),
                _format_metric(summary, "vram_mb"),
            ]
        )

        if args.verbose:
            print(f"  [{backbone_repo} / {backbone_device} :: {codec_repo} / {codec_device}]")
            for idx, result in enumerate(results, start=1):
                providers_display = ", ".join(result.providers or [])
                rtf = result.realtime_factor if result.realtime_factor is not None else float("nan")
                vram = result.vram_mb if result.vram_mb is not None else float("nan")
                print(
                    "    run {idx}: load={load:.3f}s, infer={infer:.3f}s, total={total:.3f}s, "
                    "rtf={rtf:.3f}, ram={ram:.3f}MB, vram={vram:.3f}MB, providers={providers}".format(
                        idx=idx,
                        load=result.load_s,
                        infer=result.inference_s,
                        total=result.total_s,
                        rtf=rtf,
                        ram=result.ram_mb,
                        vram=vram,
                        providers=providers_display or "n/a",
                    )
                )

    table = ""
    if rows:
        table = _render_table(headers, rows)
        print(table)
    else:
        print("No successful benchmark runs.")

    if args.summary_output:
        summary_lines = [
            "System information:",
            *metadata_lines,
            "",
            "Benchmark summary:",
            "",
            table if rows else "No data",
        ]
        args.summary_output.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.output:
        _store_results(args.output, grouped_results, summaries)


if __name__ == "__main__":
    main(list(sys.argv[1:]))
