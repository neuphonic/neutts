import os
import torch
import numpy as np
import pytest
from neutts import NeuTTS, BACKBONE_LANGUAGE_MAP


_ALL_BACKBONES = list(BACKBONE_LANGUAGE_MAP.keys())
_QUICK_BACKBONES = [
    "neuphonic/neutts-air",
    "neuphonic/neutts-air-q4-gguf",
]
_SLOW_BACKBONES = [b for b in _ALL_BACKBONES if b not in _QUICK_BACKBONES]
_SLOW_GGUF_BACKBONES = [b for b in _SLOW_BACKBONES if b.endswith("gguf")]
_QUICK_GGUF_BACKBONES = [b for b in _QUICK_BACKBONES if b.endswith("gguf")]

CODECS = [
    "neuphonic/neucodec",
    "neuphonic/distill-neucodec",
    "neuphonic/neucodec-onnx-decoder",
]


@pytest.fixture()
def reference_data() -> tuple[torch.Tensor, str]:
    ref_codes = torch.load("./samples/dave.pt")
    with open("./samples/dave.txt", "r") as f:
        ref_text = f.read()
    return ref_codes, ref_text


def _run_inference_test(backbone, codec, reference_data):
    """Loads a backbone+codec pair and validates the audio output."""
    ref_codes, ref_text = reference_data
    try:
        model = NeuTTS(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu",
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")

    audio = model.infer(text="Testing.", ref_codes=ref_codes, ref_text=ref_text)

    assert isinstance(audio, np.ndarray), "Output should be a numpy array"
    assert len(audio) > 0, "Generated audio should not be empty"
    assert not np.isnan(audio).any(), "Audio contains NaN values"
    assert audio.dtype in [np.float32, np.float64]

    print(f"Successfully generated {len(audio) / 24000:.2f}s of audio for {codec}")


def _run_streaming_test(backbone, codec, reference_data):
    """Loads a backbone+codec pair and validates streaming output."""
    ref_codes, ref_text = reference_data
    try:
        model = NeuTTS(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu",
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")

    gen = model.infer_stream(
        "This is a streaming test that should be comprised of multiple chunks.",
        ref_codes,
        ref_text,
    )

    chunks = []
    for chunk in gen:
        assert isinstance(chunk, np.ndarray)
        chunks.append(chunk)

    assert len(chunks) > 0, "Stream yielded no audio chunks"


@pytest.mark.parametrize("backbone", _QUICK_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_model_loading_and_inference(backbone, codec, reference_data):
    _run_inference_test(backbone, codec, reference_data)


@pytest.mark.parametrize("backbone", _SLOW_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_model_loading_and_inference_slow(backbone, codec, reference_data):
    if "RUN_SLOW" not in os.environ:
        pytest.skip("Skipping slow tests...")
    else:
        _run_inference_test(backbone, codec, reference_data)


@pytest.mark.parametrize("backbone", _QUICK_GGUF_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_streaming_ggml(backbone, codec, reference_data):
    _run_streaming_test(backbone, codec, reference_data)


@pytest.mark.parametrize("backbone", _SLOW_GGUF_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_streaming_ggml_slow(backbone, codec, reference_data):
    if "RUN_SLOW" not in os.environ:
        pytest.skip("Skipping slow tests...")
    else:
        _run_streaming_test(backbone, codec, reference_data)
