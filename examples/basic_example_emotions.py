import soundfile as sf
from neutts import NeuTTS2E


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
    output_path="output.wav",
):
    tts = NeuTTS2E(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo=codec,
        codec_device=codec_device,
        seed=seed,
    )

    print(f"Generating '{emotion}' audio for speaker '{speaker}': {input_text}")
    wav = tts.infer(
        input_text, speaker=speaker, emotion=emotion, temperature=temperature, top_k=top_k
    )

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS-2E Example")
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
        default="neuphonic/neutts-2e",
        help="Huggingface repo or local path containing the backbone checkpoint",
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
        default="neuphonic/neucodec",
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
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
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
        output_path=args.output_path,
    )
