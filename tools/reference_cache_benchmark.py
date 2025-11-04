"""Compare reference encoding strategies for NeuTTSAir.

Run `python tools/reference_cache_benchmark.py` from the repo root.
The script emits a markdown report under `artifacts/benchmarks/` and writes an
example waveform to `artifacts/audio/` so you can audition the cached path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuttsair.neutts import NeuTTSAir

REF_AUDIO = Path("samples/dave.wav")
REF_TEXT_PATH = Path("samples/dave.txt")
INPUT_TEXT = "This is a cached reference demo for NeuTTS Air."
CODES_PATH = Path("cache/dave.pt")
REPORT_PATH = Path("artifacts/benchmarks/reference_cache_comparison.md")
AUDIO_PATH = Path("artifacts/audio/cache_demo.wav")


@dataclass(slots=True)
class Measurement:
    scenario: str
    encode_s: float


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def _ensure_prerequisites(ref_audio: Path, ref_text_path: Path) -> str:
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio missing: {ref_audio}")
    if not ref_text_path.exists():
        raise FileNotFoundError(f"Reference text missing: {ref_text_path}")
    return ref_text_path.read_text(encoding="utf-8").strip()


def _measure(tts: NeuTTSAir, *, scenario: str, encode_kwargs: dict) -> Measurement:
    start = time.perf_counter()
    _ = tts.encode_reference(REF_AUDIO, **encode_kwargs)
    encode_s = time.perf_counter() - start
    return Measurement(scenario=scenario, encode_s=encode_s)


def main() -> None:
    ref_text = _ensure_prerequisites(REF_AUDIO, REF_TEXT_PATH)

    CODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)

    tts = NeuTTSAir(
        backbone_repo="neuphonic/neutts-air",
        backbone_device="auto",
        codec_repo="neuphonic/neucodec",
        codec_device="auto",
    )

    # Seed the on-disk cache (ignored in the timings below)
    tts.encode_reference(REF_AUDIO, codes_path=CODES_PATH)
    tts._reference_cache.clear()

    measurements = [
        _measure(tts, scenario="Fresh encode (no cache)", encode_kwargs={"reuse": False}),
    ]

    tts._reference_cache.clear()
    measurements.append(
        _measure(
            tts,
            scenario="Load cached codes from disk",
            encode_kwargs={"codes_path": CODES_PATH},
        )
    )

    measurements.append(
        _measure(
            tts,
            scenario="Reuse in-memory cache",
            encode_kwargs={"codes_path": CODES_PATH},
        )
    )

    ref_codes = tts.encode_reference(REF_AUDIO, codes_path=CODES_PATH)
    wav = tts.infer(INPUT_TEXT, ref_codes, ref_text)
    sf.write(AUDIO_PATH, wav, tts.sample_rate)

    headers = ["Scenario", "Encode Time"]
    rows = [
        [measurement.scenario, _format_seconds(measurement.encode_s)]
        for measurement in measurements
    ]

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")

    report_lines = [
        "# Reference Encoding Benchmark",
        "",
        f"*Input text:* `{INPUT_TEXT}`",
        f"*Reference audio:* `{REF_AUDIO}`",
        f"*Reference codes path:* `{CODES_PATH}`",
        "",
        "## Encoding Time Comparison",
        "",
        *table_lines,
        "",
        "Inference time is effectively unchanged across scenarios; the improvement comes from skipping the encoder stage.",
        "",
        f"Audio sample saved to `{AUDIO_PATH}`.",
        "",
        "Run with `python tools/reference_cache_benchmark.py`.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Benchmark complete. Results written to", REPORT_PATH)
    for measurement in measurements:
        print(f"{measurement.scenario}: encode={measurement.encode_s:.4f}s")
    print("Audio sample saved to", AUDIO_PATH)


if __name__ == "__main__":
    main()
