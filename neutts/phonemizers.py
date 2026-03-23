from typing import Union, List
from phonemizer.backend import EspeakBackend
import platform
import glob
import os
from pathlib import Path


def _configure_espeak_library() -> bool:
    """Configure phonemizer to use the espeak-ng bundled with this package.

    Falls back to system/Homebrew espeak-ng if the bundled version is not present
    (e.g. when running from a source checkout without building).

    Returns True if the bundled version was loaded, False if a system fallback was used.
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        pkg_dir = Path(__file__).parent

        # Locate the bundled shared library
        system = platform.system()
        if system == "Windows":
            patterns = ["espeak-ng*.dll"]
        elif system == "Darwin":
            patterns = ["libespeak-ng*.dylib"]
        else:
            patterns = ["libespeak-ng.so*", "libespeak-ng*.so"]

        lib_path = None
        for pattern in patterns:
            matches = list(pkg_dir.glob(pattern))
            if matches:
                lib_path = str(matches[0])
                break

        if lib_path:
            EspeakWrapper.set_library(lib_path)

            # Point espeak-ng at the bundled data directory
            data_dir = pkg_dir / "espeak-ng-data"
            if data_dir.exists():
                os.environ["ESPEAK_DATA_PATH"] = str(data_dir)
            return True

    except Exception:
        pass

    # Fallback 1: look for the bundled library in the active venv's site-packages.
    # This handles running pytest from a source checkout whose venv contains the
    # installed neutts wheel (which ships libespeak-ng).
    try:
        import site
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        system = platform.system()
        if system == "Windows":
            lib_pattern = "espeak-ng*.dll"
        elif system == "Darwin":
            lib_pattern = "libespeak-ng*.dylib"
        else:
            lib_pattern = "libespeak-ng*.so*"

        for site_dir in site.getsitepackages():
            for candidate in Path(site_dir).glob(f"neutts/{lib_pattern}"):
                EspeakWrapper.set_library(str(candidate))
                data_dir = candidate.parent / "espeak-ng-data"
                if data_dir.exists():
                    os.environ["ESPEAK_DATA_PATH"] = str(data_dir)
                return True
    except Exception:
        pass

    # Fallback 2: search common Homebrew/system paths on macOS
    if platform.system() == "Darwin":
        search_paths = [
            "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
            "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
            "/opt/homebrew/Cellar/espeak/*/lib/libespeak.*.dylib",
            "/usr/local/Cellar/espeak/*/lib/libespeak.*.dylib",
        ]
        for pattern in search_paths:
            matches = glob.glob(pattern)
            if matches:
                try:
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                    EspeakWrapper.set_library(matches[0])
                except Exception:
                    pass
                break

    return False


# Call before using phonemizer. Tracks whether we loaded the bundled espeak-ng.
_using_bundled_espeak = _configure_espeak_library()


class BasePhonemizer:

    def __init__(self, language_code: str = None):
        self.code = language_code
        if not self.code:
            raise ValueError(
                "A language code must be provided either via argument or subclass default"
            )

        self.g2p = EspeakBackend(
            language=self.code,
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            language_switch="remove-flags",
        )

        self.espeak_version = self.g2p.version()  # returns (major, minor, patch)

        if not _using_bundled_espeak:
            version_str = ".".join(str(v) for v in self.espeak_version)
            print(
                f"\nWARNING: You are using espeak-ng version {version_str}, which is not the "
                "supported version bundled with NeuTTS. This version is not supported and may "
                "not work as intended, particularly for non-English languages. "
                "To use the correct version, reinstall the package via pip: pip install neutts\n"
            )

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
