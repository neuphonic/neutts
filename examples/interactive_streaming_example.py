"""Interactive NeuTTSAir streaming demo that reuses precomputed reference codes.

Run from the repo root:

    python -m examples.interactive_streaming_example --ref-codes samples/dave.pt --ref-text samples/dave.txt

Type text into the prompt and the audio will stream out in real time. Press Ctrl+C
or enter an empty line to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyaudio
import torch

from neuttsair.neutts import NeuTTSAir


def _read_reference_text(ref_text: str | Path | None) -> str:
    if not ref_text:
        raise ValueError("A reference text string or file path is required for streaming.")

    candidate = Path(ref_text)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()

    return str(ref_text)


def _load_reference_codes(codes_path: Path) -> list[int]:
    if not codes_path.exists():
        raise FileNotFoundError(
            f"Reference codes not found at {codes_path}. Use `examples/encode_reference.py`"
            " to generate them first."
        )

    tensor = torch.load(codes_path, map_location="cpu")
    normalized = NeuTTSAir._as_reference_tensor(tensor)
    return normalized.to(torch.long).tolist()


def _open_audio_stream(sample_rate: int) -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    return pa, stream


def _play_chunks(
    tts: NeuTTSAir,
    *,
    text: str,
    ref_codes: list[int],
    ref_text: str,
    stream: pyaudio.Stream,
    buffer_seconds: float,
) -> None:
    buffer_samples = max(int(buffer_seconds * tts.sample_rate), 1)
    pending = np.empty(0, dtype=np.float32)

    for chunk in tts.infer_stream(text, ref_codes, ref_text):
        chunk = np.asarray(chunk, dtype=np.float32)
        pending = np.concatenate((pending, chunk))

        while pending.size >= buffer_samples:
            to_play = pending[:buffer_samples]
            audio = np.clip(to_play, -1.0, 1.0)
            stream.write((audio * 32767).astype(np.int16).tobytes())
            pending = pending[buffer_samples:]

    if pending.size:
        audio = np.clip(pending, -1.0, 1.0)
        stream.write((audio * 32767).astype(np.int16).tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive NeuTTSAir streaming demo")
    parser.add_argument(
        "--backbone",
        default="neuphonic/neutts-air-q8-gguf",
        help="GGUF backbone repo id (streaming requires llama.cpp exports).",
    )
    parser.add_argument(
        "--backbone-device",
        default="auto",
        help="Preferred device for the backbone (auto/cpu/gpu/cuda).",
    )
    parser.add_argument(
        "--codec-repo",
        default="neuphonic/neucodec-onnx-decoder",
        help="Codec backend to load (e.g. neucodec, distill-neucodec, neucodec-onnx-decoder).",
    )
    parser.add_argument(
        "--codec-device",
        default="auto",
        help="Codec device/provider hint. For ONNX: auto/cuda/directml/etc. For torch: cpu/cuda/mps.",
    )
    parser.add_argument(
        "--ref-codes",
        default="samples/dave.pt",
        type=Path,
        help="Path to precomputed reference codes (generated via encode_reference).",
    )
    parser.add_argument(
        "--ref-text",
        default="samples/dave.txt",
        help="Reference text string or path to a text file matching the reference voice.",
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=25,
        help="Number of frames per streamed chunk (higher delays playback but reduces chatter).",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=1.5,
        help="Seconds of audio to buffer before writing to the speaker.",
    )
    args = parser.parse_args(argv)

    if not args.backbone.endswith("gguf"):
        raise SystemExit("Streaming only supports GGUF backbones (llama.cpp exports).")

    ref_codes = _load_reference_codes(Path(args.ref_codes))
    ref_text = _read_reference_text(args.ref_text)

    print("Loading NeuTTSAir…")
    tts = NeuTTSAir(
        backbone_repo=args.backbone,
        backbone_device=args.backbone_device,
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
    )

    args.frames_per_chunk = max(1, args.frames_per_chunk)
    tts.streaming_frames_per_chunk = args.frames_per_chunk
    tts.streaming_stride_samples = tts.streaming_frames_per_chunk * tts.hop_length

    pa, stream = _open_audio_stream(tts.sample_rate)
    print("Ready. Enter text to synthesise (blank line to exit).")

    try:
        while True:
            try:
                text = input("neuTTS> ").strip()
            except EOFError:
                print()
                break

            if not text:
                break

            print("Streaming…")
            _play_chunks(
                tts,
                text=text,
                ref_codes=ref_codes,
                ref_text=ref_text,
                stream=stream,
                buffer_seconds=args.buffer_seconds,
            )
            print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
