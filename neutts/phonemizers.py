from typing import Union, List
from phonemizer.backend import EspeakBackend


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
    

CUSTOM_PHONEMIZERS = {
    "fr-fr": FrenchPhonemizer(),
}