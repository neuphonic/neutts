import os
from importlib import import_module

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


class _DummyCodec:
    loaded_from = None

    @classmethod
    def from_pretrained(cls, codec_repo):
        cls.loaded_from = codec_repo
        return cls()

    def eval(self):
        self.evaluated = True
        return self

    def to(self, device):
        self.device = device
        return self


class _DummyNeuCodec(_DummyCodec):
    pass


class _DummyDistillNeuCodec(_DummyCodec):
    pass


class _DummyOnnxDecoder:
    loaded_from = None
    loaded_file = None

    def __init__(self, codec_repo):
        type(self).loaded_file = codec_repo

    @classmethod
    def from_pretrained(cls, codec_repo):
        cls.loaded_from = codec_repo
        return cls(codec_repo)


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


@pytest.mark.parametrize(
    "architecture,dummy_codec",
    [
        ("NeuCodec", _DummyNeuCodec),
        ("DistillNeuCodec", _DummyDistillNeuCodec),
    ],
)
def test_load_codec_supports_local_model_directories(
    tmp_path, monkeypatch, architecture, dummy_codec
):
    neutts_module = import_module("neutts.neutts")
    monkeypatch.setattr(neutts_module, "NeuCodec", _DummyNeuCodec)
    monkeypatch.setattr(neutts_module, "DistillNeuCodec", _DummyDistillNeuCodec)
    _DummyNeuCodec.loaded_from = None
    _DummyDistillNeuCodec.loaded_from = None

    codec_dir = tmp_path / "snapshot"
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text(f'{{"architectures": ["{architecture}"]}}')

    model = NeuTTS.__new__(NeuTTS)
    model._is_onnx_codec = False
    model._load_codec(str(codec_dir), "cpu")

    assert dummy_codec.loaded_from == str(codec_dir)
    assert model.codec.evaluated is True
    assert model.codec.device == "cpu"
    assert model._is_onnx_codec is False


def test_load_codec_supports_local_onnx_file_without_fallthrough(tmp_path, monkeypatch):
    neucodec = import_module("neucodec")
    monkeypatch.setattr(neucodec, "NeuCodecOnnxDecoder", _DummyOnnxDecoder, raising=False)
    _DummyOnnxDecoder.loaded_file = None

    codec_file = tmp_path / "decoder.onnx"
    codec_file.write_bytes(b"")

    model = NeuTTS.__new__(NeuTTS)
    model._is_onnx_codec = False
    model._load_codec(str(codec_file), "cpu")

    assert _DummyOnnxDecoder.loaded_file == str(codec_file)
    assert model._is_onnx_codec is True


def test_load_codec_supports_local_onnx_directories(tmp_path, monkeypatch):
    neucodec = import_module("neucodec")
    monkeypatch.setattr(neucodec, "NeuCodecOnnxDecoder", _DummyOnnxDecoder, raising=False)
    _DummyOnnxDecoder.loaded_from = None

    codec_dir = tmp_path / "neucodec-onnx-decoder"
    codec_dir.mkdir()
    (codec_dir / "decoder.onnx").write_bytes(b"")

    model = NeuTTS.__new__(NeuTTS)
    model._is_onnx_codec = False
    model._load_codec(str(codec_dir), "cpu")

    assert _DummyOnnxDecoder.loaded_from == str(codec_dir)
    assert model._is_onnx_codec is True


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
