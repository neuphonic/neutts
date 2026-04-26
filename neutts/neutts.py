import os
import random
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

# Compiled once at import; reused by every _decode / streaming call.
_SPEECH_TOKEN_RE = re.compile(r"<\|speech_(\d+)\|>")


def _normalize_device(device: str) -> str:
    """
    Resolve a device string to a canonical torch device name.

    Accepted values:
      "auto"   – picks CUDA > MPS (Apple Metal) > CPU
      "gpu"    – legacy alias: picks CUDA > MPS > CPU
      "metal"  – explicit Apple Metal alias, maps to "mps"
      "mps"    – Apple Metal Performance Shaders
      "cuda"   – NVIDIA CUDA
      "cpu"    – CPU
    """
    device = device.lower().strip()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "metal":
        return "mps"
    if device == "gpu":
        # Legacy GGUF alias — prefer CUDA, fall back to MPS then CPU
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


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


def _linear_overlap_add(
    frames: list[np.ndarray], stride: int, power: float = 1.0
) -> np.ndarray:
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

        # Resolved device strings (canonical torch names)
        self._backbone_device = _normalize_device(backbone_device)
        self._codec_device = _normalize_device(codec_device)

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        self._load_phonemizer(language, backbone_repo)

        self._load_backbone(backbone_repo, self._backbone_device)

        self._load_codec(codec_repo, self._codec_device)

        # Load watermarker (optional)
        try:
            import perth

            self.watermarker = perth.PerthImplicitWatermarker()
        except (ImportError, AttributeError, TypeError) as e:
            warnings.warn(
                f"Perth watermarking unavailable: {e}. "
                "Audio will not be watermarked. "
                "Install with: pip install perth>=0.2.0"
            )
            self.watermarker = None

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
            self.phonemizer = CUSTOM_PHONEMIZERS[language]
        else:
            self.phonemizer = BasePhonemizer(language_code=language)

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python\n"
                    "For Apple Metal (macOS), compile with:\n"
                    "    CMAKE_ARGS='-DGGML_METAL=ON' pip install llama-cpp-python"
                ) from e

            seed = random.randint(0, 2**32)
            print(f"Using seed {seed}")

            # Metal (MPS) and CUDA both offload all layers to the GPU.
            # Flash attention is CUDA-only — it must NOT be enabled on Metal.
            use_gpu_layers = backbone_device in ("cuda", "mps")
            use_flash_attn = backbone_device == "cuda"

            if os.path.isfile(backbone_repo):
                self.backbone = Llama(
                    model_path=backbone_repo,
                    verbose=False,
                    n_gpu_layers=-1 if use_gpu_layers else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=use_flash_attn,
                    seed=seed,
                )
            else:
                self.backbone = Llama.from_pretrained(
                    repo_id=backbone_repo,
                    filename="*.gguf",
                    verbose=False,
                    n_gpu_layers=-1 if use_gpu_layers else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=use_flash_attn,
                    seed=seed,
                )

            self._is_quantized_model = True

        else:
            # Use float16 on GPU backends for lower memory and faster inference.
            # MPS (Apple Metal) supports float16; bfloat16 is not reliably supported.
            if backbone_device in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(
                backbone_repo, torch_dtype=dtype
            ).to(torch.device(backbone_device))

            # Cache special token IDs and the constant prompt template so
            # _apply_chat_template and _infer_torch pay zero tokenizer
            # overhead per inference call.
            self._tok_speech_end = self.tokenizer.convert_tokens_to_ids(
                "<|SPEECH_GENERATION_END|>"
            )
            self._tok_speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
            self._tok_speech_gen_start = self.tokenizer.convert_tokens_to_ids(
                "<|SPEECH_GENERATION_START|>"
            )
            self._tok_text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
            self._tok_text_prompt_start = self.tokenizer.convert_tokens_to_ids(
                "<|TEXT_PROMPT_START|>"
            )
            self._tok_text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
            _chat = (
                "user: Convert the text to speech:<|TEXT_REPLACE|>\n"
                "assistant:<|SPEECH_REPLACE|>"
            )
            self._prompt_template_ids = self.tokenizer.encode(_chat)

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")

        if codec_repo.endswith(".onnx") and os.path.isfile(codec_repo):
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import NeuCodecOnnxDecoder. "
                    "Make sure `neucodec` and `onnxruntime` are installed."
                ) from e

            self.codec = NeuCodecOnnxDecoder(codec_repo)
            self._is_onnx_codec = True

        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder" | "neuphonic/neucodec-onnx-decoder-int8":

                if codec_device not in ("cpu", "mps"):
                    raise ValueError(
                        "The ONNX decoder supports 'cpu' and 'mps' (Apple Metal) only. "
                        "For NVIDIA GPU inference use the standard neucodec codec with codec_device='cuda'."
                    )

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

                if codec_device == "mps":
                    try:
                        import onnxruntime as ort
                        available = ort.get_available_providers()
                        if "CoreMLExecutionProvider" in available:
                            warnings.warn(
                                "CoreML execution provider is available. "
                                "Re-instantiate the ONNX session manually with "
                                "providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'] "
                                "for Neural Engine acceleration."
                            )
                    except ImportError:
                        pass

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    def infer(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio.
            temperature (float): Sampling temperature. Lower = more deterministic.
            top_k (int): Top-k sampling. 0 disables top-k filtering.
        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text, temperature, top_k)
        else:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
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
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from
            text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio.
            temperature (float): Sampling temperature. Lower = more deterministic.
            top_k (int): Top-k sampling. 0 disables top-k filtering.
        Yields:
            np.ndarray: Generated speech waveform.
        """

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text, temperature, top_k)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        if not self._is_onnx_codec:
            wav_tensor = wav_tensor.to(self.codec.device)
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str) -> np.ndarray:
        speech_ids = [int(m) for m in _SPEECH_TOKEN_RE.findall(codes)]
        if not speech_ids:
            raise ValueError("No valid speech tokens found in the output.")
        return self._decode_from_ids(speech_ids)

    def _decode_from_ids(self, speech_ids: list[int]) -> np.ndarray:
        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                    self.codec.device
                )
                recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        ids = list(self._prompt_template_ids)

        text_replace_idx = ids.index(self._tok_text_replace)
        ids = (
            ids[:text_replace_idx]
            + [self._tok_text_prompt_start]
            + input_ids
            + [self._tok_text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(self._tok_speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [self._tok_speech_gen_start] + list(codes)

        return ids

    def _infer_torch(
        self, prompt_ids: list[int], temperature: float = 1.0, top_k: int = 50
    ) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_new_tokens=self.max_context,
                eos_token_id=self._tok_speech_end,
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

    def _infer_ggml(
        self,
        ref_codes: list[int],
        ref_text: str,
        input_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(
        self,
        ref_codes: torch.Tensor,
        ref_text: str,
        input_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        # Store raw integer codes rather than token strings — avoids "".join()
        # + regex on every chunk decode; slicing integers is O(1) per element.
        int_cache: list[int] = list(ref_codes)
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=temperature,
            top_k=top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,
        ):
            output_str = item["choices"][0]["text"]
            m = _SPEECH_TOKEN_RE.search(output_str)
            if m:
                int_cache.append(int(m.group(1)))

            if (
                len(int_cache) - n_decoded_tokens
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
                recon = self._decode_from_ids(int_cache[tokens_start:tokens_end])
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

                # Watermark the actual yielded slice, not each overlapping window.
                if self.watermarker is not None:
                    processed_recon = self.watermarker.apply_watermark(
                        processed_recon, sample_rate=24_000
                    )
                yield processed_recon

        # final decoding handled separately as non-constant chunk size
        remaining_tokens = len(int_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(
                len(int_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0,
            )
            sample_start = (
                len(int_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            recon = self._decode_from_ids(int_cache[tokens_start:])
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]

            if self.watermarker is not None:
                processed_recon = self.watermarker.apply_watermark(
                    processed_recon, sample_rate=24_000
                )
            yield processed_recon
