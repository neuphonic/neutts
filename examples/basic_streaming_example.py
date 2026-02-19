import os
import torch
import numpy as np
from neutts import NeuTTS
import pyaudio
import time


def _read_if_path(value: str) -> str:
    return open(value, "r", encoding="utf-8").read().strip() if os.path.exists(value) else value


def main(input_text, ref_codes_path, ref_text, backbone):

    assert backbone in [
        "neuphonic/neutts-air-q4-gguf",
        "neuphonic/neutts-air-q8-gguf",
        "neuphonic/neutts-nano-q4-gguf",
        "neuphonic/neutts-nano-q8-gguf",
        "neuphonic/neutts-nano-french-q4-gguf",
        "neuphonic/neutts-nano-french-q8-gguf",
        "neuphonic/neutts-nano-spanish-q4-gguf",
        "neuphonic/neutts-nano-spanish-q8-gguf",
        "neuphonic/neutts-nano-german-q4-gguf",
        "neuphonic/neutts-nano-german-q8-gguf",
    ], "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."

    # Initialize NeuTTS with the desired model and codec
    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu",
    )

    input_text = _read_if_path(input_text)
    ref_text = _read_if_path(ref_text)

    ref_codes = None
    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=tts.sample_rate,
        output=True,
        frames_per_buffer=int(tts.streaming_stride_samples),
    )

    total_audio_samples = 0
    total_lm_time = 0.0
    chunk_count = 0
    last_yield_time = None
    start_time = time.perf_counter()
    print("Streaming...")
    print("-" * 80)

    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        chunk_count += 1
        now = time.perf_counter()
        lm_duration = None
        if last_yield_time is not None:
            lm_duration = now - last_yield_time
            total_lm_time += lm_duration
        last_yield_time = now
        # Write audio
        audio = (chunk * 32767).astype(np.int16)
        stream.write(audio.tobytes(), exception_on_underflow=False)
        total_audio_samples += audio.shape[0]
        # Per-chunk timing log for latency info
        chunk_ms_actual = audio.shape[0] / tts.sample_rate * 1000
        lm_ms = f"{lm_duration * 1000:6.1f}ms" if lm_duration is not None else "  n/a "
        rt_percent = (
            (lm_duration / (chunk_ms_actual / 1000) * 100) if lm_duration is not None else 0.0
        )
        print(
            f"Chunk {chunk_count:2d}: "
            f"LM={lm_ms} │ Audio={chunk_ms_actual:5.1f}ms │ {rt_percent:5.1f}% RT"
        )
    # Add a tail pad to avoid cutting off any final generation.
    tail_pad = np.zeros(int(0.25 * tts.sample_rate), dtype=np.int16)
    stream.write(tail_pad.tobytes(), exception_on_underflow=False)
    time.sleep(0.05)

    total_time = time.perf_counter() - start_time
    total_audio_seconds = total_audio_samples / tts.sample_rate if total_audio_samples else 0.0

    # Print stats
    print("-" * 80)
    print(
        f"Streaming complete. Generated {total_audio_seconds:.2f}s of audio in {total_time:.2f}s."
    )

    if chunk_count:
        print(f"  → Average Speech LM time per chunk: {(total_lm_time / chunk_count) * 1000:.1f}ms")
        if total_audio_seconds:
            rtf = total_time / total_audio_seconds
            print(f"  → Real-Time Factor (RTF): {rtf:.2f}")

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS Example")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be converted to speech",
    )
    parser.add_argument(
        "--ref_codes",
        type=str,
        default="./samples/jo.pt",
        help="Path to pre-encoded reference audio",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/jo.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-nano-q8-gguf",
        help="Huggingface repo containing the backbone checkpoint. Must be GGUF.",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
    )
