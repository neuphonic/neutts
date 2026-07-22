import numpy as np
from neutts import NeuTTS2E
import pyaudio
import queue
import threading

try:
    from .basic_streaming_example import audio_player_thread
except ImportError:
    from basic_streaming_example import audio_player_thread


def main(
    input_text,
    speaker,
    emotion,
    backbone,
    device,
    codec,
    codec_device,
    seed,
    temperature,
    top_k,
):
    assert backbone.lower().endswith(
        "gguf"
    ), "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."

    tts = NeuTTS2E(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo=codec,
        codec_device=codec_device,
        seed=seed,
    )

    print(f"Streaming '{emotion}' audio for speaker '{speaker}': {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=tts.sample_rate, output=True)

    audio_queue = queue.Queue()
    player = threading.Thread(target=audio_player_thread, args=(audio_queue, stream))
    player.start()

    for chunk in tts.infer_stream(
        input_text, speaker=speaker, emotion=emotion, temperature=temperature, top_k=top_k
    ):
        audio = (chunk * 32767).astype(np.int16)
        audio_queue.put(audio.tobytes())

    tail_pad = np.zeros(int(0.25 * tts.sample_rate), dtype=np.int16)
    audio_queue.put(tail_pad.tobytes())
    audio_queue.put(None)
    player.join()

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS-2E Streaming Example")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be converted to speech",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="emily",
        choices=list(NeuTTS2E.SPEAKERS),
        help="One of the bundled speakers",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        choices=list(NeuTTS2E.EMOTIONS),
        help="Emotion to generate with",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-2e-q8-gguf",
        help="Huggingface repo or local path containing the backbone checkpoint. Must be GGUF.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the backbone, e.g. cpu, mps, cuda",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="neuphonic/neucodec-onnx-decoder",
        help="Huggingface repo containing the codec checkpoint",
    )
    parser.add_argument(
        "--codec_device",
        type=str,
        default="cpu",
        help="Device for the codec, e.g. cpu, mps, cuda (onnx codecs are cpu only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K sampling cutoff",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        speaker=args.speaker,
        emotion=args.emotion,
        backbone=args.backbone,
        device=args.device,
        codec=args.codec,
        codec_device=args.codec_device,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
    )
