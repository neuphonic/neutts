import os
import torch
import numpy as np
from neutts import NeuTTS
import pyaudio


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

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24_000, output=True)
    print("Streaming...")
    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        audio = (chunk * 32767).astype(np.int16)
        print(audio.shape)
        stream.write(audio.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS Example")
    parser.add_argument(
        "--input_text", type=str, required=True, help="Input text to be converted to speech"
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
        "--output_path", type=str, default="output.wav", help="Path to save the output audio"
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
