import os
import torch
import numpy as np
from neutts import NeuTTS
import pyaudio
import time
import queue
import threading


def _read_if_path(value: str) -> str:
    return open(value, "r", encoding="utf-8").read().strip() if os.path.exists(value) else value


def audio_player_thread(audio_queue, stream, prefill_chunks=0):
    # Increase prefill_chunks if RTF is slow to allow for smooth playback
    PLAYBACK_CHUNK_BYTES = 2048
    buffer = []

    for _ in range(prefill_chunks):
        chunk = audio_queue.get()
        if chunk is None:
            buffer.append(None)
            break
        buffer.append(chunk)

    for chunk in buffer:
        if chunk is None:
            audio_queue.task_done()
            return
        for i in range(0, len(chunk), PLAYBACK_CHUNK_BYTES):
            stream.write(
                chunk[i : i + PLAYBACK_CHUNK_BYTES], exception_on_underflow=False
            )
        audio_queue.task_done()

    while True:
        audio_bytes = audio_queue.get()
        if audio_bytes is None:
            audio_queue.task_done()
            break

        for i in range(0, len(audio_bytes), PLAYBACK_CHUNK_BYTES):
            slice_bytes = audio_bytes[i : i + PLAYBACK_CHUNK_BYTES]
            stream.write(slice_bytes, exception_on_underflow=False)

        audio_queue.task_done()


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
        format=pyaudio.paInt16, channels=1, rate=tts.sample_rate, output=True
    )

    audio_queue = queue.Queue()
    player = threading.Thread(target=audio_player_thread, args=(audio_queue, stream))
    player.start()

    total_audio_samples = 0
    total_gen_time = 0.0
    chunk_count = 0
    start_time = time.perf_counter()
    last_yield_time = start_time

    print("Streaming...")
    print("-" * 80)

    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        chunk_count += 1
        now = time.perf_counter()
        gen_duration = now - last_yield_time
        total_gen_time += gen_duration
        last_yield_time = now

        # Write audio
        audio = (chunk * 32767).astype(np.int16)
        audio_queue.put(audio.tobytes())
        total_audio_samples += audio.shape[0]

        # Per-chunk timing log for latency info
        chunk_ms_actual = audio.shape[0] / tts.sample_rate * 1000
        gen_ms = f"{gen_duration * 1000:6.1f}ms"
        rt_percent = gen_duration / (chunk_ms_actual / 1000) * 100

        if chunk_count == 1:
            print(
                f"Chunk {chunk_count:2d}: Generation Time={gen_ms} (TTFA) │ Chunk Size={chunk_ms_actual:5.1f}ms │ {rt_percent:5.1f}% RT"
            )
        else:
            print(
                f"Chunk {chunk_count:2d}: Generation Time={gen_ms}        │ Chunk Size={chunk_ms_actual:5.1f}ms │ {rt_percent:5.1f}% RT"
            )

    total_time = time.perf_counter() - start_time

    # Add a tail pad to avoid cutting off any final generation.
    tail_pad = np.zeros(int(0.25 * tts.sample_rate), dtype=np.int16)
    audio_queue.put(tail_pad.tobytes())

    audio_queue.put(None)
    player.join()

    total_audio_seconds = total_audio_samples / tts.sample_rate if total_audio_samples else 0.0

    # Print stats
    print("-" * 80)
    print(
        f"Streaming complete. Generated {total_audio_seconds:.2f}s of audio in {total_time:.2f}s."
    )

    if chunk_count:
        print(
            f"  → Average generation time per chunk: {(total_gen_time / chunk_count) * 1000:.1f}ms"
        )
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
