from typing import Generator
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import warnings
import perth
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTSAir:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="auto",
        codec_repo="neuphonic/neucodec",
        codec_device="auto",
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 32768
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

        # Load phonemizer + models
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        resolved_backbone_device = self._select_backbone_device(backbone_repo, backbone_device)
        self.backbone_device = resolved_backbone_device
        self._load_backbone(backbone_repo, resolved_backbone_device)

        self._requested_codec_device = codec_device or "auto"
        self.codec_device = None
        self._load_codec(codec_repo, self._requested_codec_device)

        # Load watermarker
        self.watermarker = perth.PerthImplicitWatermarker()

    @staticmethod
    def _is_mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    @classmethod
    def _select_torch_device(cls, requested: str | None) -> str:
        if requested is None:
            requested_str = "auto"
        else:
            requested_str = str(requested)

        normalized = requested_str.lower()

        if normalized in {"auto", "default"}:
            if torch.cuda.is_available():
                return "cuda"
            if cls._is_mps_available():
                return "mps"
            return "cpu"

        if normalized == "gpu":
            requested_str = "cuda"
            normalized = "cuda"

        if normalized.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available; falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "cpu"

        if normalized == "mps" and not cls._is_mps_available():
            warnings.warn(
                "MPS requested but not available; falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "cpu"

        return requested_str

    @classmethod
    def _select_backbone_device(cls, backbone_repo: str, requested: str | None) -> str:
        requested = requested or "auto"
        normalized = str(requested).lower()

        if backbone_repo.endswith("gguf"):
            if normalized in {"auto", "gpu", "cuda"}:
                # Prefer CUDA when available, otherwise prefer Apple MPS on macOS
                if torch.cuda.is_available():
                    return "gpu"
                if cls._is_mps_available():
                    # Some llama.cpp builds for macOS support Metal; prefer using
                    # an 'mps' signal so downstream loading logic can enable
                    # appropriate offload behaviour.
                    return "mps"
                warnings.warn(
                    "GPU-backed GGUF requested but no supported GPU provider was detected; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return "cpu"
            if normalized == "cpu":
                return "cpu"
            raise ValueError(
                "Unsupported backbone_device for GGUF backbones. "
                "Expected one of {'auto', 'cpu', 'gpu', 'cuda'}."
            )

        resolved = cls._select_torch_device(requested)
        return resolved

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        # GGUF loading
        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                # For GGUF/llama-cpp we set `n_gpu_layers` to -1 to enable
                # GPU-backed layers when the selected device is a GPU. We
                # also accept `mps` for macOS builds that provide Metal
                # acceleration; treat `mps` equivalently to `gpu` for the
                # purposes of n_gpu_layers but avoid enabling CUDA-specific
                # flags like flash_attn.
                n_gpu_layers=-1 if backbone_device in {"gpu", "mps"} else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
            )
            self._is_quantized_model = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo, codec_device):
        match codec_repo:
            case "neuphonic/neucodec":
                resolved_device = self._select_torch_device(codec_device)
                print(f"Loading codec from: {codec_repo} on {resolved_device} ...")
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(resolved_device)
                self.codec_device = resolved_device
            case "neuphonic/distill-neucodec":
                resolved_device = self._select_torch_device(codec_device)
                print(f"Loading codec from: {codec_repo} on {resolved_device} ...")
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(resolved_device)
                self.codec_device = resolved_device
            case "neuphonic/neucodec-onnx-decoder":
                normalized_device = codec_device or "auto"
                print(f"Loading codec from: {codec_repo} on {normalized_device} ...")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

                self._configure_onnx_codec_session(codec_device)
                self.codec_device = normalized_device

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    def _configure_onnx_codec_session(self, codec_device: str):
        """Configure ONNX Runtime providers based on the requested device."""

        normalized_device = (codec_device or "cpu").lower()

        # Map legacy/alias device names
        if normalized_device in {"onnx", "auto"}:
            normalized_device = "auto"
        elif normalized_device in {"gpu"}:
            normalized_device = "cuda"

        device_id: str | None = None
        if ":" in normalized_device:
            base_device, device_id = normalized_device.split(":", 1)
            normalized_device = base_device

        try:
            import onnxruntime as ort
        except ImportError as e:
            if normalized_device not in {"cpu", "auto"}:
                raise ImportError(
                    "onnxruntime with the desired execution provider is not installed. "
                    "Install `onnxruntime-gpu` for CUDA or `onnxruntime-directml` for DirectML."
                ) from e
            # CPU fallback when onnxruntime-gpu isn't available
            warnings.warn(
                "onnxruntime-gpu not installed; falling back to CPUExecutionProvider.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        available = ort.get_available_providers()

        provider_priority: list[str] = []
        provider_options: list[dict[str, str]] = []

        def add_provider(provider: str, options: dict[str, str] | None = None):
            provider_priority.append(provider)
            provider_options.append(options or {})

        gpu_providers = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "MetalExecutionProvider",
            "CoreMLExecutionProvider",
        ]

        if normalized_device == "cpu":
            add_provider("CPUExecutionProvider")
        elif normalized_device == "cuda":
            provider_name = "CUDAExecutionProvider"
            if provider_name in available:
                options = {"device_id": device_id} if device_id is not None else None
                add_provider(provider_name, options)
                add_provider("CPUExecutionProvider")
            else:
                warnings.warn(
                    "CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                add_provider("CPUExecutionProvider")
        elif normalized_device in {"directml", "dml"}:
            provider_name = "DmlExecutionProvider"
            if provider_name in available:
                add_provider(provider_name)
                add_provider("CPUExecutionProvider")
            else:
                warnings.warn(
                    "DmlExecutionProvider unavailable; falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                add_provider("CPUExecutionProvider")
        elif normalized_device in {"metal"}:
            provider_name = "MetalExecutionProvider"
            if provider_name in available:
                add_provider(provider_name)
                add_provider("CPUExecutionProvider")
            else:
                warnings.warn(
                    "MetalExecutionProvider unavailable; falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                add_provider("CPUExecutionProvider")
        elif normalized_device in {"coreml"}:
            provider_name = "CoreMLExecutionProvider"
            if provider_name in available:
                add_provider(provider_name)
                add_provider("CPUExecutionProvider")
            else:
                warnings.warn(
                    "CoreMLExecutionProvider unavailable; falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                add_provider("CPUExecutionProvider")
        elif normalized_device == "rocm":
            provider_name = "ROCMExecutionProvider"
            if provider_name in available:
                add_provider(provider_name)
                add_provider("CPUExecutionProvider")
            else:
                warnings.warn(
                    "ROCMExecutionProvider unavailable; falling back to CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                add_provider("CPUExecutionProvider")
        elif normalized_device == "auto":
            for provider_name in gpu_providers:
                if provider_name in available:
                    add_provider(provider_name)
                    break
            if not provider_priority:
                warnings.warn(
                    "No GPU execution providers available; using CPUExecutionProvider.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            add_provider("CPUExecutionProvider")
        else:
            raise ValueError(
                "Unsupported codec_device for ONNX decoder. "
                "Expected one of {'cpu', 'auto', 'cuda', 'cuda:<id>', 'gpu', 'directml', 'dml', 'rocm', 'onnx'}."
            )

        # Filter out providers that truly aren't available (ignoring CPU which always works)
        filtered_priority = []
        filtered_options = []
        for provider_name, options in zip(provider_priority, provider_options, strict=False):
            if provider_name == "CPUExecutionProvider" or provider_name in available:
                filtered_priority.append(provider_name)
                filtered_options.append(options)

        if not filtered_priority:
            raise RuntimeError("No valid ONNX Runtime providers available to initialize the codec.")

        try:
            self.codec.session.set_providers(filtered_priority, filtered_options)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to configure ONNX Runtime providers: {filtered_priority}."
            ) from exc

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            output_str = self._infer_torch(prompt_ids)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)

        return watermarked_wav
    
    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Yields:
            np.ndarray: Generated speech waveform.
        """ 

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

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
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
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

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str
    
    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
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
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(self, ref_codes: torch.Tensor, ref_text: str, input_text: str) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (
                    n_decoded_tokens - tokens_start
                ) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[
                    n_decoded_samples:new_samples_end
                ]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 
                0
            )
            sample_start = (
                len(token_cache) 
                - tokens_start 
                - remaining_tokens 
                - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon