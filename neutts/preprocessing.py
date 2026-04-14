import unicodedata
import jieba
from pypinyin import lazy_pinyin, Style
from pypinyin_dict.phrase_pinyin_data import large_pinyin
import pyopenjtalk


class ZHFrontend:

    def __init__(self):
        large_pinyin.load()
        self.PUNCT = frozenset(",.!?:;")

    def __call__(self, text: str) -> str:
        """Chinese G2P pipeline: NFKC normalization, punctuation mapping, and pinyin via pypinyin."""
        if not text.strip():
            return ""

        # NFKC normalizes full-width alphanumerics and punctuation
        text = unicodedata.normalize("NFKC", text)
        # Map CJK punct that NFKC doesn't cover
        # and strip quotes/brackets (complicated to handle)
        text = text.translate(str.maketrans("、。", ",.", "«»《》「」【】\"'()[]{}"))

        result = []
        for seg in jieba.cut(text):
            pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=False)
            for syllable in pinyin:
                syllable = syllable.strip()
                if not syllable:
                    continue

                if result:
                    # Space between syllables; no space before prosodic punctuation
                    if syllable not in self.PUNCT:
                        result.append(" ")

                result.append(syllable)

        return "".join(result)


class JAFrontend:

    def __init__(self):
        self._KATA2HIRA = str.maketrans(
            {chr(k): chr(k - 96) for k in range(0x30A1, 0x30F7)}
        )

    def _kata2hira(self, text: str) -> str:
        return text.translate(self._KATA2HIRA)

    def _is_katakana(self, text: str) -> bool:
        return all("\u30a0" <= c <= "\u30ff" or c == "ー" for c in text)

    def __call__(self, text: str) -> str:
        """Japanese G2P pipeline: NFKC normalization and hiragana via pyopenjtalk."""
        if not text.strip():
            return ""

        # NFKC normalizes full-width alphanumerics
        text = unicodedata.normalize("NFKC", text)

        result = []
        for word in pyopenjtalk.run_frontend(text):
            pron, mora_size = word["pron"], word["mora_size"]

            if mora_size > 0:
                # Strip accent marker (\u2019) which can appear mid-word
                pron = pron.replace("\u2019", "")
                # Keep katakana if original surface was katakana (loanwords etc.)
                if self._is_katakana(word["string"]):
                    result.append(pron)
                elif word["string"] in ("は", "へ"):
                    result.append(word["string"])
                else:
                    result.append(self._kata2hira(pron))
            else:
                # Keep original punctuation
                surface = word["string"].strip()
                if surface:
                    result.append(surface)

        return "".join(result)
