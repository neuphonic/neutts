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
    default_code = None

    def __init__(self, language_code: str = None):
        self.code = language_code or self.default_code
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
    default_code = "fr-fr"
    def clean(self, phonemes: str) -> str:
        # Remove dashes (common in french output - indicates syllable, but not needed)
        return phonemes.replace("-", "")


class GermanPhonemizer(BasePhonemizer):
    default_code = "de"

    def clean(self, phonemes: str) -> str:
        if (1, 50, 0) <= self.espeak_version < (1, 52, 0):
            warnings.warn(
                "espeak-ng versions between 1.50.0 and 1.51.1 have a German phonemization issue (https://github.com/espeak-ng/espeak-ng/issues/890). Attempting to fix, but consider upgrading espeak-ng to 1.52.0 or later if possible."
            )
            # Patch a german phonemization issue present in these espeak versions
            # See https://github.com/espeak-ng/espeak-ng/issues/890 and the fix in 1.52.0
            # https://github.com/espeak-ng/espeak-ng/commit/c517074825422bfc7c2400f74ff4b4fb3d96e26e
            original = phonemes
            phonemes = re.sub(
                r"y(?!ː)", "ʏ", phonemes
            )  # Replace 'y' with 'ʏ' when not followed by a length marker
            phonemes = phonemes.replace("i???", "iɐʊɐ")
            phonemes = phonemes.replace("??", "ʊɐ")
            phonemes = phonemes.replace("i?", "iɐ")
            if phonemes != original:
                warnings.warn(
                    f"Attempted to fix German phonemization issue. Before: {original} After: {phonemes}"
                )
            if "?" in phonemes:  # should be an extremely rare edge case
                warnings.warn(
                    "Attempted fix failed. Please consider upgrading espeak-ng to 1.52.0 or later."
                )
        return phonemes


CUSTOM_PHONEMIZERS = {
    "fr-fr": FrenchPhonemizer(),
    "de": GermanPhonemizer(),
}
