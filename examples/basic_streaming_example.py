import os
import torch
import numpy as np
from neutts import NeuTTS
import pyaudio
import time


def _read_if_path(value: str) -> str:
    return open(value, "r", encoding="utf-8").read().strip() if os.path.exists(value) else value


def main(input_text, ref_codes_path, ref_text, backbone, language):

    # assert backbone in [
    #     "neuphonic/neutts-air-q4-gguf",
    #     "neuphonic/neutts-air-q8-gguf",
    #     "neuphonic/neutts-nano-q4-gguf",
    #     "neuphonic/neutts-nano-q8-gguf",
    #     "neuphonic/neutts-nano-french-q4-gguf",
    #     "neuphonic/neutts-nano-french-q8-gguf",
    #     "neuphonic/neutts-nano-spanish-q4-gguf",
    #     "neuphonic/neutts-nano-spanish-q8-gguf",
    #     "neuphonic/neutts-nano-german-q4-gguf",
    #     "neuphonic/neutts-nano-german-q8-gguf",
    # ], "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."

    # Initialize NeuTTS with the desired model and codec
    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu",
        language=language,
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
    print("Streaming...")

    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        # Write audio
        audio = (chunk * 32767).astype(np.int16)
        stream.write(audio.tobytes(), exception_on_underflow=False)
        total_audio_samples += audio.shape[0]

    # Add a tail pad to avoid cutting off any final generation.
    tail_pad = np.zeros(int(0.5 * tts.sample_rate), dtype=np.int16)
    stream.write(tail_pad.tobytes(), exception_on_underflow=False)
    time.sleep(0.05)

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
        default="./samples/carla.pt",
        help="Path to pre-encoded reference audio",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/carla.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="NeuphonicClients/neutts-nano-german-update-q4-gguf",
        help="Huggingface repo containing the backbone checkpoint. Must be GGUF.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Espeak language code for phonemization",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
        language=args.language,
    )
