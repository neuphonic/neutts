from pathlib import Path

import numpy as np
import torch

from .neutts import NeuTTS

# Wheel installs bundle the references inside the package (see CMakeLists.txt);
# source checkouts read them from the repo's samples directory.
_BUNDLED_SAMPLES = Path(__file__).parent / "samples"


class NeuTTS2E(NeuTTS):
    """NeuTTS-2E: fixed-speaker emotional English TTS with pre-encoded references."""

    SAMPLE_DIR = (
        _BUNDLED_SAMPLES if _BUNDLED_SAMPLES.exists() else Path(__file__).parents[1] / "samples"
    )
    SPEAKERS = ("emily", "paul", "sophie", "steven")
    EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-2e",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        seed=None,
    ):
        super().__init__(
            backbone_repo=backbone_repo,
            backbone_device=backbone_device,
            codec_repo=codec_repo,
            codec_device=codec_device,
            seed=seed,
        )
        if self._supported_emotions is None:
            self._supported_emotions = list(self.EMOTIONS)
        self._speaker_refs = {}

    def _speaker(self, name: str) -> tuple[torch.Tensor, str]:
        if name not in self.SPEAKERS:
            raise ValueError(f"Unknown speaker '{name}'. Available speakers: {list(self.SPEAKERS)}")
        if name not in self._speaker_refs:
            codes = torch.load(self.SAMPLE_DIR / f"{name}.pt")
            text = (self.SAMPLE_DIR / f"{name}.txt").read_text().strip()
            self._speaker_refs[name] = (codes, text)
        return self._speaker_refs[name]

    def infer(
        self,
        text: str,
        speaker: str = "emily",
        emotion: str = "neutral",
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        ref_codes, ref_text = self._speaker(speaker)
        return super().infer(
            text, ref_codes, ref_text, emotion=emotion, temperature=temperature, top_k=top_k
        )

    def infer_stream(
        self,
        text: str,
        speaker: str = "emily",
        emotion: str = "neutral",
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        ref_codes, ref_text = self._speaker(speaker)
        return super().infer_stream(
            text, ref_codes, ref_text, emotion=emotion, temperature=temperature, top_k=top_k
        )

    def warmup(self) -> None:
        if self._is_quantized_model:
            for _ in self.infer_stream("Warming up."):
                pass
        else:
            self.infer("Warming up.")
