import os
import re
import random
import warnings
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torchaudio
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM

from .phonemizers import BasePhonemizer, CUSTOM_PHONEMIZERS


# Maps known Neuphonic backbone repos to their eSpeak language code.
# Used to auto-select a phonemizer when `language` is not explicitly provided.
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


class NeuTTS:
    """Neural text-to-speech model combining a language model backbone with a neural codec.

    Supports both PyTorch (full-precision) and GGUF (quantised, via llama-cpp-python)
    backbone variants. Streaming inference is available for GGUF models only.
    """

    # Fallback Jinja2 chat template for GGUF models that do not embed their
    # own template in the GGUF metadata. 
    _DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}user: Convert the text to speech:<|TEXT_PROMPT_START|>{{ message['content'] }}<|TEXT_PROMPT_END|>"
        "{% elif message['role'] == 'assistant' %}\nassistant:<|SPEECH_GENERATION_START|>{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}\nassistant:{% endif %}"
    )

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        language=None,
    ):
        """Initialise NeuTTS by loading the backbone, codec, phonemizer, and watermarker.

        Args:
            backbone_repo: HuggingFace repo ID or local path to the backbone checkpoint.
                           Paths ending in ``.gguf`` are loaded via llama-cpp-python;
                           all others are loaded via ``transformers``.
            backbone_device: Device for the backbone (``"cpu"`` or ``"gpu"``).
            codec_repo: HuggingFace repo ID or local ``.onnx`` path for the neural codec.
            codec_device: Device for the codec (``"cpu"`` or ``"cuda"``).
            language: eSpeak language code (e.g. ``"en-us"``, ``"de"``).
                      Required when ``backbone_repo`` is not in ``BACKBONE_LANGUAGE_MAP``
                      and the model uses phoneme input.
        """
        # Streaming / decoding constants
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

        # Load phonemizer + models
        print("Loading phonemizer...")
        # Call before using phonemizer 
        # TODO: Modify branching condition
        if "qwen3" not in backbone_repo:
            self._load_phonemizer(language, backbone_repo)

        self._backbone_repo = backbone_repo
        
        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

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


    def _load_phonemizer(self, language: str | None, backbone_repo: str) -> None:
        """Load the phonemizer for the given language.

        Args:
            language: eSpeak language code, or ``None`` to infer from ``backbone_repo``.
            backbone_repo: Used for language look-up when ``language`` is ``None``.
        Raises:
            ValueError: If ``language`` cannot be determined.
        """
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
    
    def _load_backbone(self, backbone_repo: str, backbone_device: str) -> None:
        """Load the LLM backbone — either a GGUF model or a HuggingFace model.

        GGUF models (repo ID / path ending in ``gguf``) are loaded via
        llama-cpp-python and support CPU/GPU offloading and streaming inference.
        All other repos are loaded via ``transformers``.

        Args:
            backbone_repo: HuggingFace repo ID or local file path.
            backbone_device: ``"cpu"`` or ``"gpu"``.
        """
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            seed = random.randint(0, 2**32)
            print(f"Using seed {seed}")

            if os.path.isfile(backbone_repo):
                self.backbone = Llama(
                    model_path=backbone_repo,
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                    seed=seed,
                )
            else:
                self.backbone = Llama.from_pretrained(
                    repo_id=backbone_repo,
                    filename="*.gguf",
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                    seed=seed,
                )

            self._is_quantized_model = True
            self._setup_ggml_template()

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo: str, codec_device: str) -> None:
        """Load the neural codec used to encode and decode speech tokens.

        Supports the following repos:
        - ``neuphonic/neucodec``
        - ``neuphonic/distill-neucodec``
        - ``neuphonic/neucodec-onnx-decoder`` / ``neuphonic/neucodec-onnx-decoder-int8``
        - A local ``.onnx`` file path.

        Args:
            codec_repo: HuggingFace repo ID or local ``.onnx`` file path.
            codec_device: ``"cpu"`` or ``"cuda"``.
        Raises:
            ValueError: If ``codec_repo`` is not recognised.
        """
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

                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    def _to_phones(self, text: str) -> str:
        """Convert text to a space-separated phoneme string via the loaded phonemizer.

        Args:
            text: Raw input text.
        Returns:
            str: Space-separated phoneme sequence.
        """
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        return " ".join(phones)

    def _setup_ggml_template(self) -> None:
        """Compile the Jinja2 chat template used to format GGML prompts.

        Reads the template from the GGUF model metadata if present; otherwise
        falls back to ``_DEFAULT_CHAT_TEMPLATE``. Also resolves the BOS/EOS
        token strings from the metadata for use inside the template.

        Results are stored as:
        - ``self._ggml_template``: compiled Jinja2 template object.
        - ``self._ggml_bos_token`` / ``self._ggml_eos_token``: token strings.
        """
        import jinja2

        template_str = self.backbone.metadata.get(
            "tokenizer.chat_template", self._DEFAULT_CHAT_TEMPLATE
        )

        bos_id = self.backbone.metadata.get("tokenizer.ggml.bos_token_id")
        eos_id = self.backbone.metadata.get("tokenizer.ggml.eos_token_id")
        self._ggml_bos_token = self.backbone.detokenize([int(bos_id)]).decode("utf-8") if bos_id else ""
        self._ggml_eos_token = self.backbone.detokenize([int(eos_id)]).decode("utf-8") if eos_id else ""

        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._ggml_template = env.from_string(template_str)

    def _format_ggml_prompt(self, user_content: str, assistant_prefix: str) -> str:
        """Build a raw completion prompt for GGML inference.

        Renders the chat template with only the user message and
        ``add_generation_prompt=True``, then appends the assistant prefix
        directly. This leaves the prompt mid-assistant-turn so that raw
        completion continues from the prefix rather than starting a new turn.

        Args:
            user_content: The user message content (may include special tokens).
            assistant_prefix: Text to place at the start of the assistant turn
                              (e.g. the reference speech codes).
        Returns:
            str: Formatted prompt ready to pass to the GGML backbone.
        """
        user_prompt = self._ggml_template.render(
            messages=[{"role": "user", "content": user_content}],
            bos_token=self._ggml_bos_token,
            eos_token=self._ggml_eos_token,
            add_generation_prompt=True,
        )
        return user_prompt + assistant_prefix

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> str:
        """Build a prompt for torch inference using the HuggingFace chat template.

        For phoneme-based models the texts are first converted to phonemes and
        a simple hand-crafted template is used. For newer models the
        tokenizer's own ``apply_chat_template`` is called.

        Args:
            ref_codes: Reference speech token indices.
            ref_text: Text corresponding to the reference audio.
            input_text: Text to synthesise.
        Returns:
            str: Formatted prompt string (not tokenised).
        """
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])

        # TODO: Modify branching condition
        if "qwen3" not in self._backbone_repo:
            # use old template
            ref_text = self._to_phones(ref_text)
            input_text = self._to_phones(input_text)
            prompt = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"""

        else:
            messages = [{"role": "user", "content": f"{ref_text} {input_text}"}, {"role": "assistant", "content": codes_str}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        return prompt

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> np.ndarray:
        """Synthesise speech from text, conditioned on a reference voice.

        Args:
            text: Input text to convert to speech.
            ref_codes: Encoded reference audio produced by :meth:`encode`.
            ref_text: Transcript of the reference audio.
        Returns:
            np.ndarray: 1-D float32 waveform at ``self.sample_rate`` Hz.
        """
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text)
        else:
            output_str = self._infer_torch(ref_codes, ref_text, text)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = (
            wav
            if self.watermarker is None
            else self.watermarker.apply_watermark(wav, sample_rate=24_000)
        )

        return watermarked_wav

    def infer_stream(
        self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str
    ) -> Generator[np.ndarray, None, None]:
        """Synthesise speech with streaming output, yielding audio chunks as they are decoded.

        Only supported for GGUF (quantised) backbone models.

        Args:
            text: Input text to convert to speech.
            ref_codes: Encoded reference audio produced by :meth:`encode`.
            ref_text: Transcript of the reference audio.
        Yields:
            np.ndarray: 1-D float32 audio chunks at ``self.sample_rate`` Hz.
        Raises:
            NotImplementedError: If called with a torch (non-GGUF) backbone.
        """
        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)

        raise NotImplementedError("Streaming is not implemented for the torch backend!")

    @torch.no_grad()
    def encode(self, ref_audio_path: str | Path) -> torch.Tensor:
        """Encode a reference audio file into discrete speech token indices.

        The audio is resampled to 16 kHz and converted to mono before encoding.

        Args:
            ref_audio_path: Path to the reference audio file (any format supported
                            by ``torchaudio``).
        Returns:
            torch.Tensor: 1-D tensor of integer speech token indices.
        """
        wav, sr = torchaudio.load(ref_audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav_tensor = wav.float().unsqueeze(0)  # [1, 1, T]
        ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes
    
    def encode_reference(self, ref_audio_path: str | Path) -> torch.Tensor:
        """Alias for :meth:`encode` kept for backwards compatibility."""
        return self.encode(ref_audio_path)


    def _decode(self, codes: str) -> np.ndarray:
        """Decode a string of speech tokens into a waveform.

        Args:
            codes: String containing speech tokens of the form ``<|speech_N|>``.
        Returns:
            np.ndarray: 1-D float32 audio waveform.
        Raises:
            ValueError: If no valid speech tokens are found in ``codes``.
        """
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if not speech_ids:
            raise ValueError("No valid speech tokens found in the output.")

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

    def _infer_torch(self, ref_codes: list[int], ref_text: str, text: str) -> str:
        """Run a single forward pass through the torch backbone.

        Args:
            ref_codes: Reference speech token indices.
            ref_text: Transcript of the reference audio.
            text: Input text to synthesise.
        Returns:
            str: Raw model output containing generated speech tokens.
        """
        prompt = self._apply_chat_template(ref_codes, ref_text, text)
        prompt_ids = self.tokenizer.encode(prompt)
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
        """Run a single forward pass through the GGML backbone.

        Args:
            ref_codes: Reference speech token indices.
            ref_text: Transcript of the reference audio.
            input_text: Input text to synthesise.
        Returns:
            str: Raw model output containing generated speech tokens.
        """
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])

        # TODO: Modify branching condition
        if "qwen3" not in self._backbone_repo:
            ref_text = self._to_phones(ref_text)
            input_text = self._to_phones(input_text)

        prompt = self._format_ggml_prompt(
            user_content=f"{ref_text} {input_text}",
            assistant_prefix=f"{codes_str}",
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        return output["choices"][0]["text"]

    def _linear_overlap_add(
        self, frames: list[np.ndarray], power: float = 1.0
    ) -> np.ndarray:
        """Overlap-add a list of audio frames into a single waveform.

        Each frame is weighted by a triangular envelope and accumulated at
        intervals of ``self.streaming_stride_samples``. Adjacent frames overlap
        so that the envelope weights sum to a constant, avoiding discontinuities.

        Original implementation:
            https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py

        Args:
            frames: List of 1-D (or N-D with time as the last axis) float32 arrays.
            power: Exponent applied to the triangular weight (1.0 = linear taper).
        Returns:
            np.ndarray: Reconstructed waveform of shape ``(*frame.shape[:-1], total_samples)``.
        """
        stride = self.streaming_stride_samples
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

    def _infer_stream_ggml(
        self, ref_codes: torch.Tensor, ref_text: str, input_text: str
    ) -> Generator[np.ndarray, None, None]:
        """Stream speech generation through the GGML backbone, yielding decoded audio chunks.

        Speech tokens are accumulated in a rolling cache. Each time enough new
        tokens arrive (``streaming_frames_per_chunk + streaming_lookforward``),
        a chunk is decoded with overlap-add smoothing and yielded. Any
        remaining tokens are flushed as a final chunk after generation ends.

        Args:
            ref_codes: Reference speech token indices.
            ref_text: Transcript of the reference audio.
            input_text: Input text to synthesise.
        Yields:
            np.ndarray: 1-D float32 audio chunks at ``self.sample_rate`` Hz.
        """
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])

        # TODO: Modify branching condition
        if "qwen3" not in self._backbone_repo:
            ref_text = self._to_phones(ref_text)
            input_text = self._to_phones(input_text)

        prompt = self._format_ggml_prompt(
            user_content=f"{ref_text} {input_text}",
            assistant_prefix=f"{codes_str}",
        )
        stream = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in stream:
            output_str = item["choices"][0]["text"]
            if output_str:
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
                processed_recon = self._linear_overlap_add(audio_cache)
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

            processed_recon = self._linear_overlap_add(audio_cache)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon
