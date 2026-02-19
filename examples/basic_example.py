import os
import soundfile as sf
from neutts import NeuTTS
import torch


def main(input_text, ref_audio_path, ref_text, backbone, output_path="output.wav"):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize NeuTTS with the desired model and codec
    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    )

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if not os.path.exists(ref_audio_path.replace(".wav", ".pt")):
        print("Encoding reference audio")
        ref_codes = tts.encode_reference(ref_audio_path)
        torch.save(ref_codes, ref_audio_path.replace(".wav", ".pt"))
    else:
        print("Loading pre-encoded reference audio")
        ref_codes = torch.load(ref_audio_path.replace(".wav", ".pt"))

    print(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS Example")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be converted to speech",
    )
    parser.add_argument(
        "--ref_audio", type=str, default="./samples/jo.wav", help="Path to reference audio file"
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
        default="neuphonic/neutts-nano",
        help="Huggingface repo containing the backbone checkpoint",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
    )
