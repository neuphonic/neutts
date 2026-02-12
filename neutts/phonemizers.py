from typing import Union, List
from phonemizer.backend import EspeakBackend
import platform
import glob
import warnings
import re

def _configure_espeak_library():
    """Auto-detect and configure espeak library on macOS."""
    if platform.system() != "Darwin":
        return  # Only needed on macOS

    # Common Homebrew installation paths
    search_paths = [
        "/opt/homebrew/Cellar/espeak/*/lib/libespeak.*.dylib",  # Apple Silicon
        "/usr/local/Cellar/espeak/*/lib/libespeak.*.dylib",  # Intel
        "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",  # Apple Silicon
        "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
    ]

    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper

                EspeakWrapper.set_library(matches[0])
                return
            except Exception:
                # If this fails, phonemizer will try its default detection
                pass


# Call before using phonemizer
_configure_espeak_library()


class BasePhonemizer:

    def __init__(self, language_code: str = None):
        self.code = language_code
        if not self.code:
            raise ValueError("A language code must be provided either via argument or subclass default")

        self.g2p = EspeakBackend(
            language=self.code,
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            language_switch="remove-flags",
        )

        self.espeak_version = self.g2p.version()  # returns (major, minor, patch)

    def preprocess(self, text: str) -> str:
        """Language-specific text preprocessing."""
        return text

    def clean(self, phonemes: str) -> str:
        """Language-specific phoneme cleanup."""
        return phonemes

    def phonemize(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Phonemize text (or list of texts), then clean the output."""
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        preprocessed_text = [self.preprocess(t) for t in text]
        phonemes_list = self.g2p.phonemize(preprocessed_text)
        cleaned_list = [self.clean(p) for p in phonemes_list]

        return cleaned_list[0] if single_input else cleaned_list


class FrenchPhonemizer(BasePhonemizer):

    def __init__(self, language_code: str = "fr-fr"):
        super().__init__(language_code)

    def clean(self, phonemes: str) -> str:
        # Remove dashes (common in french output - indicates syllable, but not needed)
        return phonemes.replace("-", "")


CUSTOM_PHONEMIZERS = {
    "fr-fr": FrenchPhonemizer(),
}