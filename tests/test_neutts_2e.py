import contextlib
import io
import os
import re

import numpy as np
import pytest
import torch

from neutts import NeuTTS, NeuTTS2E
from neutts.neutts import _normalize_text

GEN_TEXT = "I can't believe it's finally here!"

_ALL_BACKBONES = [
    "neuphonic/neutts-2e",
    "neuphonic/neutts-2e-q4-gguf",
    "neuphonic/neutts-2e-q8-gguf",
]
_GGUF_BACKBONES = [b for b in _ALL_BACKBONES if b.endswith("gguf")]

CODECS = [
    "neuphonic/neucodec",
    "neuphonic/distill-neucodec",
    "neuphonic/neucodec-onnx-decoder",
]


@pytest.fixture()
def speaker_data() -> tuple[str, str]:
    return "emily", "happy"


def _run_inference_test(backbone, codec, speaker_data):
    """Loads a backbone+codec pair and validates the audio output."""
    speaker, emotion = speaker_data
    try:
        model = NeuTTS2E(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu",
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")

    audio = model.infer(text="Testing.", speaker=speaker, emotion=emotion)

    assert isinstance(audio, np.ndarray), "Output should be a numpy array"
    assert len(audio) > 0, "Generated audio should not be empty"
    assert not np.isnan(audio).any(), "Audio contains NaN values"
    assert audio.dtype in [np.float32, np.float64]

    print(f"Successfully generated {len(audio) / 24000:.2f}s of audio for {codec}")


def _run_streaming_test(backbone, codec, speaker_data):
    """Loads a backbone+codec pair and validates streaming output."""
    speaker, emotion = speaker_data
    try:
        model = NeuTTS2E(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu",
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")

    gen = model.infer_stream(
        "This is a streaming test that should be comprised of multiple chunks.",
        speaker=speaker,
        emotion=emotion,
    )

    chunks = []
    for chunk in gen:
        assert isinstance(chunk, np.ndarray)
        chunks.append(chunk)

    assert len(chunks) > 0, "Stream yielded no audio chunks"


@pytest.mark.parametrize("backbone", _ALL_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_model_loading_and_inference(backbone, codec, speaker_data):
    _run_inference_test(backbone, codec, speaker_data)


@pytest.mark.parametrize("backbone", _GGUF_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_streaming_ggml(backbone, codec, speaker_data):
    _run_streaming_test(backbone, codec, speaker_data)


def test_speaker_data_complete():
    assert NeuTTS2E.SPEAKERS == ("emily", "paul", "sophie", "steven")
    for name in NeuTTS2E.SPEAKERS:
        assert (NeuTTS2E.SAMPLE_DIR / f"{name}.wav").exists()
        assert (NeuTTS2E.SAMPLE_DIR / f"{name}.txt").read_text().strip()
        codes = torch.load(NeuTTS2E.SAMPLE_DIR / f"{name}.pt")
        assert codes.ndim == 1 and len(codes) > 0


@pytest.fixture(scope="module")
def tts() -> NeuTTS2E:
    return NeuTTS2E()


@pytest.fixture(scope="module")
def gguf_tts() -> NeuTTS2E:
    return NeuTTS2E(
        backbone_repo="neuphonic/neutts-2e-q4-gguf",
        codec_repo="neuphonic/neucodec-onnx-decoder",
    )


def test_all_speakers_resolve(tts):
    for name in tts.SPEAKERS:
        ref_codes, ref_text = tts._speaker(name)
        assert len(ref_codes) > 0 and ref_text


def test_supported_emotions_loaded(tts):
    assert sorted(tts._supported_emotions) == sorted(NeuTTS2E.EMOTIONS)
    assert tts._check_emotion("neutral") is None
    with pytest.raises(ValueError, match="Unknown emotion"):
        tts._check_emotion("furious")


def test_unknown_speaker_raises(tts):
    with pytest.raises(ValueError, match="Unknown speaker"):
        tts.infer("Testing.", speaker="dave")


def test_unknown_emotion_raises(tts):
    with pytest.raises(ValueError, match="Unknown emotion"):
        tts.infer("Testing.", speaker="emily", emotion="furious")


def test_normalize_text():
    assert _normalize_text("“Don’t.”") == '"Don\'t."'


@pytest.mark.parametrize("emotion", ["neutral", "happy"])
def test_prompt_emotion(tts, emotion):
    tokenizer = tts.tokenizer
    ref_codes, ref_text = tts._speaker("emily")
    ids = tts._apply_chat_template(ref_codes, ref_text, GEN_TEXT, tts._check_emotion(emotion))
    codes_str = "".join(f"<|speech_{i}|>" for i in ref_codes.tolist())
    if emotion == "neutral":
        text_section = f"{ref_text} {GEN_TEXT}"
    else:
        text_section = f"{ref_text}<|{emotion.upper()}|>{GEN_TEXT}"
    assert tokenizer.decode(ids) == (
        f"<|TEXT_PROMPT_START|>{text_section}<|TEXT_PROMPT_END|>"
        f"<|SPEECH_GENERATION_START|>{codes_str}"
    )

    if emotion != "neutral":
        # Ref and gen text must be encoded separately either side of the emotion token.
        emotion_id = tokenizer.convert_tokens_to_ids(f"<|{emotion.upper()}|>")
        idx = ids.index(emotion_id)
        start = ids.index(tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>"))
        assert ids[start + 1 : idx] == tokenizer.encode(ref_text, add_special_tokens=False)


def test_prompt_unknown_emotion_raises(tts):
    ref_codes, ref_text = tts._speaker("emily")
    with pytest.raises(ValueError, match="not in the model vocab"):
        tts._apply_chat_template(ref_codes, ref_text, GEN_TEXT, emotion="ecstatic")


@pytest.mark.parametrize("emotion", ["neutral", "happy"])
def test_ggml_emotion_prompt_matches_torch_prompt(tts, gguf_tts, emotion):
    ref_codes, ref_text = tts._speaker("emily")
    emo = tts._check_emotion(emotion)
    torch_ids = tts._apply_chat_template(ref_codes, ref_text, GEN_TEXT, emotion=emo)
    ggml_prompt = gguf_tts._ggml_prompt(ref_codes, ref_text, GEN_TEXT, emo)
    assert tts.tokenizer.decode(torch_ids) == ggml_prompt
    assert gguf_tts.backbone.tokenize(ggml_prompt.encode(), add_bos=True, special=True) == torch_ids


def test_phoneme_model_rejects_emotion():
    model = NeuTTS(
        backbone_repo="neuphonic/neutts-air-q4-gguf",
        codec_repo="neuphonic/neucodec-onnx-decoder",
    )
    ref_codes = torch.load("./samples/jo.pt")
    with open("./samples/jo.txt") as f:
        ref_text = f.read().strip()
    with pytest.raises(ValueError, match="BPE"):
        model.infer("Testing.", ref_codes, ref_text, emotion="happy")


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {},
        {
            "backbone_repo": "neuphonic/neutts-2e-q4-gguf",
            "codec_repo": "neuphonic/neucodec-onnx-decoder",
        },
    ],
    ids=["torch", "gguf"],
)
def test_seed_semantics(model_kwargs):
    if "RUN_SLOW" not in os.environ:
        pytest.skip("Skipping slow tests...")

    text = "Testing the seed semantics."
    kwargs = {"speaker": "paul", "emotion": "happy"}

    unseeded = NeuTTS2E(**model_kwargs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        takes = [unseeded.infer(text, **kwargs) for _ in range(2)]
    seeds = [int(s) for s in re.findall(r"Using seed (\d+)", buf.getvalue())]
    assert len(set(seeds)) == 2
    assert not np.array_equal(takes[0], takes[1])

    recovered = NeuTTS2E(**model_kwargs, seed=seeds[1])
    assert np.array_equal(recovered.infer(text, **kwargs), takes[1])

    seeded = NeuTTS2E(**model_kwargs, seed=42)
    assert np.array_equal(seeded.infer(text, **kwargs), seeded.infer(text, **kwargs))
