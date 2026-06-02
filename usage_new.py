import spaces
import json
import os
import sys
import subprocess
import numpy as np
import gradio as gr

# Install normaliser
GIT_TOKEN = os.environ.get("GIT_TOKEN")
NORMALISER_COMMIT = "531b835e6c1bdc0f5f7a411dcf4321a0bb897e65"
NORMALISER_REPO_URL = f"git+https://{GIT_TOKEN}@github.com/neuphonic/normaliser.git@{NORMALISER_COMMIT}#egg=normaliser"
print("Installing normaliser...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", NORMALISER_REPO_URL], check=True)
clone_path = os.path.abspath("./src/normaliser")
if clone_path not in sys.path:
    sys.path.insert(0, clone_path)

from normaliser import GermanNormaliser, EnglishNormaliser, SpanishNormaliser, FrenchNormaliser

german_normaliser = GermanNormaliser()
english_normaliser = EnglishNormaliser()
spanish_normaliser = SpanishNormaliser()
french_normaliser = FrenchNormaliser()
NORMALISERS = {
    "english": english_normaliser,
    "german": german_normaliser,
    "spanish": spanish_normaliser,
    "french": french_normaliser,
}

SPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
ESPEAK_DIR = os.path.join(SPACE_ROOT, "espeak-precompiled")
os.environ["PATH"] = f"{os.path.join(ESPEAK_DIR, 'bin')}:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(ESPEAK_DIR, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["ESPEAK_DATA_PATH"] = os.path.join(ESPEAK_DIR, "share", "espeak-ng-data")
CUSTOM_LIB_PATH = os.path.join(ESPEAK_DIR, "lib", "libespeak-ng.so")
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(CUSTOM_LIB_PATH)

sys.path.append("neutts-air")
from neuttsair.neutts import NeuTTSAir

# ==========================================
# 1. STANDARD APP CONFIGURATION
# ==========================================

SAMPLES_DIR = os.path.join(os.getcwd(), "samples")


def _load_speakers(samples_dir: str) -> dict:
    """Build SPEAKERS by scanning samples/<lang>/transcripts.json paired with audio files.

    Each language subdir must contain a transcripts.json of the form
    {"<stem>": "<ref_text>", ...} where <stem> matches a <stem>.wav or <stem>.mp3
    audio file in the same directory. Speaker names are the raw JSON keys.
    """
    speakers = {}
    if not os.path.isdir(samples_dir):
        return speakers
    for lang_entry in sorted(os.scandir(samples_dir), key=lambda e: e.name):
        if not lang_entry.is_dir():
            continue
        transcripts_path = os.path.join(lang_entry.path, "transcripts.json")
        if not os.path.isfile(transcripts_path):
            continue
        with open(transcripts_path, encoding="utf-8") as f:
            transcripts = json.load(f)
        lang_speakers = {}
        for stem, ref_text in transcripts.items():
            audio_path = None
            for ext in (".wav", ".mp3"):
                candidate = os.path.join(lang_entry.path, stem + ext)
                if os.path.isfile(candidate):
                    audio_path = candidate
                    break
            if audio_path is None:
                continue
            lang_speakers[stem] = {"ref_text": ref_text, "ref_audio": audio_path}
        if lang_speakers:
            speakers[lang_entry.name] = lang_speakers
    return speakers


# Speakers keyed by language name. Auto-populated from samples/<lang>/transcripts.json.
SPEAKERS = _load_speakers(SAMPLES_DIR)

# Example generation texts keyed by language name.
GEN_TEXTS = {
    "english": "My name is Paul. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all.",
    "spanish": "Me llamo Martina. Tengo 25 años y me acabo de mudar a Londres. El metro es un poco lioso, pero me lleva a todas partes enseguida.",
    "german": "Niedersachsen ist ein großes Bundesland im Nordwesten Deutschlands und bietet eine Vielzahl von Landschaften und Kulturen.",
    "french": "Je m'appelle Amélie. J'ai 25 ans et je viens d'emménager à Londres. Le métro est assez déroutant, mais il m'emmène partout en un rien de temps.",
    "urdu": "میرا نام جو ہے۔ میری عمر پچیس سال ہے اور میں ابھی لندن منتقل ہوا ہوں۔ انڈرگراؤنڈ کافی الجھن بھری ہے، لیکن یہ مجھے بہت کم وقت میں ادھر ادھر پہنچا دیتی ہے۔",
    "japanese": "私の名前はミワです。二十五歳で、ロンドンに引っ越してきたばかりです。地下鉄はかなり複雑ですが、あっという間にどこへでも行けます。",
    "korean": "제 이름은 시우예요. 저는 스물다섯 살이고 런던에 막 이사 왔어요. 지하철은 꽤 복잡하지만, 어디든 금방 갈 수 있어요.",
    "chinese": "我叫梅,今年二十五岁,刚搬到伦敦。这里的地铁挺复杂的,但去哪儿都很快。",
    "portuguese": "Meu nome é Diogo. Tenho vinte e cinco anos e acabei de me mudar para Londres. O metrô é um pouco confuso, mas me leva a qualquer lugar num instante.",
}

EMOTIONS = ["neutral", "angry", "sad", "happy", "surprised", "disgusted", "fearful"]

# Per-model feature flags.
MODEL_META = {
    "qwen3-0.2b-en-cfg-distilled-emotional-sft-10k-26-03-26": {"supports_emotions": True},
    "qwen3-0.2b-en-emotional-grpo-1625-08-04-26":             {"supports_emotions": True},
}

# Default speaker per model per language. Single-language models have one entry.
# Multilingual models list all supported languages — these drive the synthesis language dropdown.
DEFAULT_SPEAKERS = {
    "qwen3-0.2b-en-cfg-distilled-emotional-sft-10k-26-03-26": {"english": "paul"},
    "qwen3-0.2b-en-emotional-grpo-1625-08-04-26":             {"english": "paul"},
}

tts_dict = {
    "qwen3-0.2b-en-cfg-distilled-emotional-sft-10k-26-03-26": NeuTTSAir(
        backbone_repo="neuphonic/qwen3-0.2b-en-cfg-distilled-emotional-sft-10k-26-03-26",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
    ),
    "qwen3-0.2b-en-emotional-grpo-1625-08-04-26": NeuTTSAir(
        backbone_repo="neuphonic/qwen3-0.2b-en-emotional-grpo-1625-08-04-26",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
    ),
}

# ==========================================
# 2. GRADIO APP LOGIC
# ==========================================


@spaces.GPU()
def infer(
    ref_text: str,
    ref_audio_path: str,
    gen_text: str,
    model: str,
    synthesis_lang: str,
    temperature: float,
    top_k: int,
    emotion: str,
) -> tuple[int, np.ndarray]:

    if not ref_audio_path or not ref_text.strip():
        raise gr.Error("Please provide both Reference Text and Reference Audio.")

    normaliser = NORMALISERS.get(synthesis_lang)
    if normaliser is not None:
        ref_text = normaliser(ref_text)
        print(f"Normalised {synthesis_lang} ref text: {ref_text}")
        gen_text = normaliser(gen_text)
        print(f"Normalised {synthesis_lang} gen text: {gen_text}")

    gr.Info(f"Starting inference request for {model}!")
    gr.Info(f"Encoding reference...")

    current_tts = tts_dict[model]
    ref_codes = current_tts.encode_reference(ref_audio_path)

    gr.Info(f"Generating audio for input text: {gen_text}")
    if current_tts.input_format == "phonemes":
        print(f"Espeak version: {current_tts.phonemizer.g2p.version()}")
        print(f"Reference phones: {current_tts._to_phones(ref_text)}")
        print(f"Phones to generate: {current_tts._to_phones(gen_text)}")

    supports_emotions = MODEL_META[model]["supports_emotions"]
    is_multilingual = len(DEFAULT_SPEAKERS[model]) > 1
    wav = current_tts.infer(
        gen_text,
        ref_codes,
        ref_text,
        language=synthesis_lang if is_multilingual else None,
        temperature=float(temperature),
        top_k=top_k,
        emotion=emotion if supports_emotions else None,
    )

    return (24_000, wav)


def _speaker_ref_outputs(lang, default_spk):
    lang_speakers = SPEAKERS.get(lang, {})
    spk_choices = list(lang_speakers.keys()) + ["Custom"]
    if default_spk and default_spk in lang_speakers:
        ref_data = lang_speakers[default_spk]
        return spk_choices, default_spk, GEN_TEXTS.get(lang, ""), ref_data["ref_text"], ref_data["ref_audio"]
    return spk_choices, "Custom", GEN_TEXTS.get(lang, ""), "", None


def update_language(model):
    langs = list(DEFAULT_SPEAKERS[model].keys())
    is_multi = len(langs) > 1
    first_lang = langs[0]
    default_spk = DEFAULT_SPEAKERS[model][first_lang]
    spk_choices, spk_value, gen_text, ref_text, ref_audio = _speaker_ref_outputs(first_lang, default_spk)
    supports_emotions = MODEL_META[model]["supports_emotions"]
    return (
        gr.update(choices=spk_choices, value=spk_value),
        gen_text,
        ref_text,
        ref_audio,
        gr.update(choices=langs, value=first_lang, visible=is_multi),
        gr.update(visible=supports_emotions),
    )


def update_synthesis_language(model, synthesis_lang):
    default_spk = DEFAULT_SPEAKERS[model][synthesis_lang]
    spk_choices, spk_value, gen_text, ref_text, ref_audio = _speaker_ref_outputs(synthesis_lang, default_spk)
    return (
        gr.update(choices=spk_choices, value=spk_value),
        gen_text,
        ref_text,
        ref_audio,
    )


def update_speaker(synthesis_lang, speaker):
    if speaker == "Custom":
        return gr.update(value=""), gr.update(value=None)
    ref_data = SPEAKERS[synthesis_lang][speaker]
    return gr.update(value=ref_data["ref_text"]), gr.update(value=ref_data["ref_audio"])


_default_model = "qwen3-0.2b-en-emotional-grpo-1625-08-04-26"
_default_lang = "english"
_default_spk = DEFAULT_SPEAKERS[_default_model][_default_lang]
_default_langs = list(DEFAULT_SPEAKERS[_default_model].keys())
_default_is_multi = len(_default_langs) > 1
_, _, _default_gen_text, _default_ref_text, _default_ref_audio = _speaker_ref_outputs(_default_lang, _default_spk)

with gr.Blocks(title="NeuTTS-Nano-Emotion - COMPARISON 😂😭😲😍") as demo:
    gr.Markdown("## NeuTTS-Nano-Emotion - COMPARISON 😂😭😲😍")
    gr.Markdown("Select a model, an emotion, and a reference speaker. Use 'Custom' to provide your own reference audio and text.")

    with gr.Row():
        lang_dropdown = gr.Dropdown(
            choices=[
                "qwen3-0.2b-en-cfg-distilled-emotional-sft-10k-26-03-26",
                "qwen3-0.2b-en-emotional-grpo-1625-08-04-26",
            ],
            value=_default_model,
            label="Model",
        )
        synthesis_lang_dropdown = gr.Dropdown(
            choices=_default_langs,
            value=_default_lang,
            visible=_default_is_multi,
            label="Synthesis Language",
        )
        speaker_dropdown = gr.Dropdown(
            choices=list(SPEAKERS.get(_default_lang, {}).keys()) + ["Custom"],
            value=_default_spk,
            label="Speaker Name",
        )
        emotion_dropdown = gr.Dropdown(
            choices=EMOTIONS,
            value="neutral",
            visible=MODEL_META[_default_model]["supports_emotions"],
            label="Emotion",
        )

    with gr.Row():
        ref_text_input = gr.Textbox(
            label="Reference Text",
            value=_default_ref_text,
        )
        ref_audio_input = gr.Audio(
            type="filepath",
            label="Reference Audio",
            value=_default_ref_audio,
        )

    gen_text_input = gr.Textbox(
        label="Text to Generate", value=_default_gen_text
    )

    with gr.Accordion("Advanced Generation Settings", open=False):
        with gr.Row():
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=100.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Higher values make output more random.",
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=1000,
                value=50,
                step=1,
                label="Top-K",
                info="Limits sampling to the K most probable tokens.",
            )

    submit_btn = gr.Button("Generate Speech", variant="primary")
    output_audio = gr.Audio(type="numpy", label="Generated Speech")

    lang_dropdown.change(
        fn=update_language,
        inputs=[lang_dropdown],
        outputs=[
            speaker_dropdown,
            gen_text_input,
            ref_text_input,
            ref_audio_input,
            synthesis_lang_dropdown,
            emotion_dropdown,
        ],
    )

    synthesis_lang_dropdown.change(
        fn=update_synthesis_language,
        inputs=[lang_dropdown, synthesis_lang_dropdown],
        outputs=[speaker_dropdown, gen_text_input, ref_text_input, ref_audio_input],
    )

    speaker_dropdown.change(
        fn=update_speaker,
        inputs=[synthesis_lang_dropdown, speaker_dropdown],
        outputs=[ref_text_input, ref_audio_input],
    )

    submit_btn.click(
        fn=infer,
        inputs=[
            ref_text_input,
            ref_audio_input,
            gen_text_input,
            lang_dropdown,
            synthesis_lang_dropdown,
            temperature_slider,
            top_k_slider,
            emotion_dropdown,
        ],
        outputs=[output_audio],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=[SAMPLES_DIR], mcp_server=True, inbrowser=True)
