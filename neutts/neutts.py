import os
import random
import subprocess
import sys
import unicodedata
from typing import Generator
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import warnings
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM
from .phonemizers import BasePhonemizer, CUSTOM_PHONEMIZERS

BACKBONE_LANGUAGE_MAP = {
    # en models
    "neuphonic/neutts-air": "en-us",
    "neuphonic/neutts-air-q4-gguf": "en-us",
    "neuphonic/neutts-air-q8-gguf": "en-us",
    "neuphonic/neutts-nano": "en-us",
    "neuphonic/neutts-nano-q4-gguf": "en-us",
    "neuphonic/neutts-nano-q8-gguf": "en-us",
    # de models
    "neuphonic/neutts-nano-german": "de",
    "neuphonic/neutts-nano-german-q4-gguf": "de",
    "neuphonic/neutts-nano-german-q8-gguf": "de",
    # fr models
    "neuphonic/neutts-nano-french": "fr-fr",
    "neuphonic/neutts-nano-french-q4-gguf": "fr-fr",
    "neuphonic/neutts-nano-french-q8-gguf": "fr-fr",
    # es models
    "neuphonic/neutts-nano-spanish": "es",
    "neuphonic/neutts-nano-spanish-q4-gguf": "es",
    "neuphonic/neutts-nano-spanish-q8-gguf": "es",
}


_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.translate(_QUOTE_MAP))


def _torch_device(device, param):
    try:
        torch.empty(0, device=device)
    except (RuntimeError, AssertionError) as e:
        raise ValueError(
            f"{param} '{device}' is not a valid and available torch device "
            f"(e.g. cpu, mps, cuda): {e}"
        ) from e
    return torch.device(device)


def _n_perf_cores() -> int:
    # On Apple silicon, keep generation threads off the efficiency cores.
    if sys.platform == "darwin":
        try:
            return int(
                subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True)
            )
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return max((os.cpu_count() or 2) // 2, 1)


def _load_watermarker():
    try:
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            # resemble-perth 1.0.1 needs pkg_resources, removed from recent setuptools
            import importlib.resources
            import types

            shim = types.ModuleType("pkg_resources")
            shim.resource_filename = lambda package, resource: str(
                importlib.resources.files(package) / resource
            )
            sys.modules["pkg_resources"] = shim

        import perth

        return perth.PerthImplicitWatermarker()
    except (ImportError, AttributeError, TypeError) as e:
        warnings.warn(
            f"Perth watermarking unavailable: {e}. "
            "Audio will not be watermarked. "
            "Install with: pip install resemble-perth"
        )
        return None


def _linear_overlap_add(frames: list[np.ndarray], stride: int, power: float = 1.0) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)

    out = np.zeros((*shape, total_size), dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]

        weight = (0.5 - np.abs(t - 0.5)) ** power

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride

    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTS:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        language=None,
        seed=None,
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # ggml & onnx flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        self._seed = seed

        self._load_backbone(backbone_repo, backbone_device)

        if self.input_format == "phonemes":
            print("Loading phonemizer...")
            self._load_phonemizer(language, backbone_repo)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker (optional)
        self.watermarker = _load_watermarker()

    @staticmethod
    def _read_gguf_array_meta(model_path):
        # llama-cpp's metadata dict drops array fields; recover them with the gguf package
        try:
            from gguf import GGUFReader, GGUFValueType
        except ImportError:
            return {}
        result = {}
        for name, field in GGUFReader(model_path, "r").fields.items():
            if field.types and field.types[0] == GGUFValueType.ARRAY:
                try:
                    result[name] = field.contents()
                except Exception:
                    pass
        return result

    def _load_phonemizer(self, language, backbone_repo):
        if not language:
            if BACKBONE_LANGUAGE_MAP.get(backbone_repo) is not None:
                language = BACKBONE_LANGUAGE_MAP[backbone_repo]
            else:
                raise ValueError(
                    "If you aren't using a Neuphonic model, make sure to specify any "
                    "eSpeak language code as the `language` parameter."
                )

        if language in CUSTOM_PHONEMIZERS:
            self.phonemizer = CUSTOM_PHONEMIZERS[language]()
        else:
            self.phonemizer = BasePhonemizer(language_code=language)

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.lower().endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            use_gpu = backbone_device.lower() != "cpu"
            gguf_kwargs = dict(
                verbose=False,
                n_gpu_layers=-1 if use_gpu else 0,
                n_ctx=self.max_context,
                n_batch=512,
                n_ubatch=512,
                n_threads=_n_perf_cores(),
                n_threads_batch=os.cpu_count(),
                use_mlock=True,
                flash_attn=use_gpu,
                seed=self._seed,
            )
            if os.path.isfile(backbone_repo):
                self.backbone = Llama(model_path=backbone_repo, **gguf_kwargs)
            else:
                self.backbone = Llama.from_pretrained(
                    repo_id=backbone_repo, filename="*.gguf", **gguf_kwargs
                )

            self._is_quantized_model = True
            self.input_format = self.backbone.metadata.get("neuphonic.input_format")
            if self.input_format is None:
                template = self.backbone.metadata.get("tokenizer.chat_template", "")
                is_bpe = (
                    "<|TEXT_PROMPT_START|>" in template
                    and "Convert the text to speech" not in template
                )
                self.input_format = "BPE" if is_bpe else "phonemes"
            gguf_arrays = self._read_gguf_array_meta(self.backbone.model_path)
            self._supported_emotions = gguf_arrays.get("neuphonic.supported_emotions")

        else:
            device = _torch_device(backbone_device, "backbone_device")
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(
                backbone_repo, dtype=torch.bfloat16
            ).to(device)
            neuphonic_cfg = getattr(self.backbone.config, "neuphonic", None) or {}
            self.input_format = neuphonic_cfg.get("input_format", "phonemes")
            self._supported_emotions = neuphonic_cfg.get("supported_emotions")

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")

        if "onnx" in codec_repo:
            if codec_device != "cpu":
                raise ValueError("Onnx decoders only currently run on CPU.")

            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import onnx decoder support."
                    " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                ) from e

            if codec_repo.endswith(".onnx") and os.path.isfile(codec_repo):
                self.codec = NeuCodecOnnxDecoder(codec_repo)
            elif codec_repo in (
                "neuphonic/neucodec-onnx-decoder",
                "neuphonic/neucodec-onnx-decoder-int8",
            ):
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
            else:
                raise ValueError(
                    "Invalid codec repo! Must be one of: 'neuphonic/neucodec',"
                    " 'neuphonic/distill-neucodec', 'neuphonic/neucodec-onnx-decoder',"
                    " 'neuphonic/neucodec-onnx-decoder-int8', or a local path to a .onnx decoder."
                )
            self._is_onnx_codec = True
            return

        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(_torch_device(codec_device, "codec_device"))
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(_torch_device(codec_device, "codec_device"))
            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of: 'neuphonic/neucodec',"
                    " 'neuphonic/distill-neucodec', 'neuphonic/neucodec-onnx-decoder',"
                    " 'neuphonic/neucodec-onnx-decoder-int8', or a local path to a .onnx decoder."
                )

    def infer(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Transcript of the reference audio.
            emotion (str | None): Emotion tag, e.g. "happy". BPE models only.
            temperature (float): Sampling temperature.
            top_k (int): Top-K sampling cutoff.
        Returns:
            np.ndarray: Generated speech waveform.
        """
        emotion = self._check_emotion(emotion)

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text, emotion, temperature, top_k)
        else:
            torch.manual_seed(self._call_seed())
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text, emotion)
            output_str = self._infer_torch(prompt_ids, temperature, top_k)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = (
            wav
            if self.watermarker is None
            else self.watermarker.apply_watermark(wav, sample_rate=24_000)
        )

        return watermarked_wav

    def infer_stream(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from
            text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Transcript of the reference audio.
            emotion (str | None): Emotion tag, e.g. "happy". BPE models only.
            temperature (float): Sampling temperature.
            top_k (int): Top-K sampling cutoff.
        Yields:
            np.ndarray: Generated speech waveform.
        """
        emotion = self._check_emotion(emotion)

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text, emotion, temperature, top_k)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

    def _call_seed(self) -> int:
        # A fixed seed pins every call; otherwise each call draws a fresh seed.
        # The print makes any unseeded take recoverable by passing its seed.
        seed = self._seed if self._seed is not None else random.randint(0, 2**32)
        print(f"Using seed {seed}")
        return seed

    def _check_emotion(self, emotion: str | None) -> str | None:
        if emotion == "neutral":
            emotion = None
        if emotion is not None:
            if self.input_format == "phonemes":
                raise ValueError("Emotion is only supported by BPE models.")
            if self._supported_emotions is not None and emotion not in self._supported_emotions:
                raise ValueError(
                    f"Unknown emotion '{emotion}'. Supported emotions: {self._supported_emotions}"
                )
        return emotion

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                recon = self.codec.decode_code(codes)

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.codec.device
                    )
                    recon = self.codec.decode_code(codes).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str, emotion: str | None = None
    ) -> list[int]:

        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        if self.input_format == "phonemes":
            input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            chat = (
                "user: Convert the text to speech:<|TEXT_REPLACE|>\n" "assistant:<|SPEECH_REPLACE|>"
            )
        else:
            ref_text = _normalize_text(ref_text)
            input_text = _normalize_text(input_text)
            if emotion is None:
                # Encode the concatenation in one pass so BPE resolves the boundary cleanly.
                input_ids = self.tokenizer.encode(
                    f"{ref_text} {input_text}", add_special_tokens=False
                )
            else:
                emotion_token = f"<|{emotion.upper()}|>"
                emotion_id = self.tokenizer.convert_tokens_to_ids(emotion_token)
                if emotion_id == self.tokenizer.unk_token_id:
                    raise ValueError(f"Emotion token {emotion_token} is not in the model vocab.")
                input_ids = (
                    self.tokenizer.encode(ref_text, add_special_tokens=False)
                    + [emotion_id]
                    + self.tokenizer.encode(input_text, add_special_tokens=False)
                )
            chat = """<|TEXT_REPLACE|><|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int], temperature: float = 1.0, top_k: int = 50) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str

    def _ggml_prompt(
        self, ref_codes: list[int], ref_text: str, input_text: str, emotion: str | None = None
    ) -> str:
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])

        if self.input_format == "phonemes":
            ref_text = self._to_phones(ref_text)
            input_text = self._to_phones(input_text)
            return (
                f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
                f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
            )

        ref_text = _normalize_text(ref_text)
        input_text = _normalize_text(input_text)
        if emotion is None:
            text = f"{ref_text} {input_text}"
        else:
            emotion_token = f"<|{emotion.upper()}|>"
            tokens = self.backbone.tokenize(emotion_token.encode(), add_bos=False, special=True)
            if len(tokens) != 1:
                raise ValueError(f"Emotion token {emotion_token} is not in the model vocab.")
            text = f"{ref_text}{emotion_token}{input_text}"
        return (
            f"<|TEXT_PROMPT_START|>{text}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>{codes_str}"
        )

    def _infer_ggml(
        self,
        ref_codes: list[int],
        ref_text: str,
        input_text: str,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        prompt = self._ggml_prompt(ref_codes, ref_text, input_text, emotion)
        # a full re-prefill keeps repeated calls bit-identical
        self.backbone.reset()
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
            seed=self._call_seed(),
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(
        self,
        ref_codes: torch.Tensor,
        ref_text: str,
        input_text: str,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        prompt = self._ggml_prompt(ref_codes, ref_text, input_text, emotion)
        # a full re-prefill keeps repeated calls bit-identical
        self.backbone.reset()

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,
            seed=self._call_seed(),
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if (
                len(token_cache[n_decoded_tokens:])
                >= self.streaming_frames_per_chunk + self.streaming_lookforward
            ):

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames)
                    * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = (
                    recon
                    if self.watermarker is None
                    else self.watermarker.apply_watermark(recon, sample_rate=24_000)
                )
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0,
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = (
                recon
                if self.watermarker is None
                else self.watermarker.apply_watermark(recon, sample_rate=24_000)
            )
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon
