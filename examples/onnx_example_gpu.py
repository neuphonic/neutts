import os
import soundfile as sf
import torch
from neuttsair.neutts import NeuTTSAir


def main(input_text, ref_codes_path, ref_text, backbone, codec_device="auto", output_path="output_onnx.wav"):
    if not ref_codes_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize NeuTTSAir with the ONNX codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="auto",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device=codec_device      # codec will use requested ONNX execution provider
    )

    # Load reference text
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    # Load pre-encoded reference audio
    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir ONNX GPU Example")
    parser.add_argument(
        "--input_text", 
        type=str, 
        required=True,
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_codes", 
        type=str, 
        default="./samples/dave.pt", 
        help="Path to pre-encoded reference audio"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt", 
        help="Reference text corresponding to the reference audio"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output_onnx_gpu.wav", 
        help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="neuphonic/neutts-air", 
        help="Huggingface repo containing the backbone checkpoint"
    )
    parser.add_argument(
        "--codec_device",
        type=str,
        default="auto",
        help=(
            "ONNX execution target (e.g. 'auto', 'cpu', 'cuda', 'cuda:1', "
            "'gpu', 'directml', 'dml', 'rocm')."
        )
    )
    args = parser.parse_args()

    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
        codec_device=args.codec_device,
        output_path=args.output_path
    )
