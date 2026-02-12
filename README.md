# NeuTTS

HuggingFace 🤗:

- NeuTTS-Air (English): [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf), [Space](https://huggingface.co/spaces/neuphonic/neutts-air)

- NeuTTS-Nano Multilingual Collection:
   - NeuTTS-Nano (English): [Model](https://huggingface.co/neuphonic/neutts-nano), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-q4-gguf)
   - NeuTTS-Nano-French: [Model](https://huggingface.co/neuphonic/neutts-nano-french), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-french-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-french-q4-gguf)
   - NeuTTS-Nano-German: [Model](https://huggingface.co/neuphonic/neutts-nano-german), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-german-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-german-q4-gguf)
   - NeuTTS-Nano-Spanish: [Model](https://huggingface.co/neuphonic/neutts-nano-spanish), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-spanish-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-spanish-q4-gguf)
   - [Multilingual Space](https://huggingface.co/spaces/neuphonic/neutts-nano-multilingual-collection)

[NeuTTS-Nano Demo Video](https://github.com/user-attachments/assets/629ec5b2-4818-4fa6-987a-99fcbadc56bc)

_Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI_

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS is a collection of open source, on-device, TTS speech language models with instant voice cloning. Built off of LLM backbones, NeuTTS brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device - unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- 🗣Best-in-class realism for their size - produce natural, ultra-realistic voices that sound human, at the sweet spot between speed, size, and quality for real-world applications
- 📱Optimised for on-device deployment - quantisations provided in GGUF format, ready to run on phones, laptops, or even Raspberry Pis
- 👫Instant voice cloning - create your own speaker with as little as 3 seconds of audio
- 🚄Simple LM + codec architecture - making development and deployment simple

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affliated with Neuphonic, our github or this repo.
>
> We are on neuphonic.com only. Please be careful out there! 🙏

## Model Details

NeuTTS models are built from small LLM backbones - lightweight yet capable language models optimised for text understanding and generation - as well as a powerful combination of technologies designed for efficiency and quality:

- **Supported Languages**: English, Spanish, German, French (model-dependent)
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - our 50hz neural audio codec that achieves exceptional audio quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Quantisations available in GGUF format for efficient on-device inference
- **Responsibility**: Watermarked outputs
- **Inference Speed**: Real-time generation on mid-range devices
- **Power Consumption**: Optimised for mobile and embedded devices


|  | NeuTTS-Air | NeuTTS-Nano Models |
|---|---:|---:|
| **# Params (Active)** | ~360m | ~120m |
| **# Params (Emb + Active)** | ~552m | ~229m |
| **Cloning** | Yes | Yes |
| **License** | Apache 2.0 | NeuTTS Open License 1.0 |

## Throughput Benchmarking

These benchmarks are for the Q4_0 quantisations [neutts-air-Q4_0](https://huggingface.co/neuphonic/neutts-air-q4-gguf) and [neutts-nano-Q4_0](https://huggingface.co/neuphonic/neutts-nano-q4-gguf). Note that all models in the NeuTTS-Nano Multilingual Collection have an identical architecture, so these results should apply for any Q4_0 model in the collection. 

CPU benchmarking used [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) (from llama.cpp) to measure prefill and decode throughput at multiple context sizes. For the GPU benchmark (RTX 4090), we leverage vLLM to maximise throughput, using the [vLLM benchmark](https://docs.vllm.ai/en/stable/cli/bench/throughput/).

We include benchmarks on four devices: Galaxy A25 5G, AMD Ryzen 9HX 370, iMac M4 16GB, NVIDIA GeForce RTX 4090.


|  | NeuTTS-Air | NeuTTS-Nano |
|---|---:|---:|
| **Galaxy A25 5G (CPU only)** | 20 tokens/s | 45 tokens/s|
| **AMD Ryzen 9 HX 370 (CPU only)** | 119 tokens/s | 221 tokens/s |
| **iMAc M4 16 GB (CPU only)** | 111 tokens/s | 195 tokens/s |
| **RTX 4090** | 16194 tokens/s | 19268 tokens/s |


> [!NOTE]
>  llama-bench used 14 threads for prefill and 16 threads for decode (as configured in the benchmark run) on AMD Ryzen 9HX 370 and iMac M4 16GB, and 6 threads for each on the Galaxy A25 5G. The tokens/s reported are when having 500 prefill tokens and generating 250 output tokens.

> [!NOTE]
> Please note that these benchmarks only include the Speech Language Model and do not include the Codec which is needed for a full audio generation pipeline.

## Get Started with NeuTTS

> [!NOTE]
> We have added a [streaming example](examples/basic_streaming_example.py) using the `llama-cpp-python` library as well as a [finetuning script](examples/finetune.py). For finetuning, please refer to the [finetune guide](TRAINING.md) for more details.

1. **Install System Dependencies (required): `espeak-ng`**

> [!CAUTION]
> `espeak-ng` is an updated version of `espeak`, as of February 2026 on version 1.52.0. Older versions of `espeak` and `espeak-ng` can exhibit significant phonemisation issues, particularly for non-English languages. Updating your system version of `espeak-ng` to the latest version possible is highly recommended. 

> [!NOTE]
> `brew` on macOS Ventura and later, `apt` in Ubuntu version 25 or Debian version 13, and `choco`/`winget` on Windows, install the latest version of `espeak-ng` with the commands below. If you have a different or older operating system, you may need to install from source: see the following link https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md

   Please refer to the following link for instructions on how to install `espeak-ng`:

   https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

   ```bash
   # Mac OS
   brew install espeak-ng

   # Ubuntu/Debian
   sudo apt install espeak-ng

   # Windows install
   # via chocolatey (https://community.chocolatey.org/packages?page=1&prerelease=False&moderatorQueue=False&tags=espeak)
   choco install espeak-ng
   # via winget
   winget install -e --id eSpeak-NG.eSpeak-NG
   # via msi (need to add to path or folow the "Windows users who installed via msi" below)
   # find the msi at https://github.com/espeak-ng/espeak-ng/releases
   ```

   Windows users who installed via msi / do not have their install on path need to run the following (see https://github.com/bootphon/phonemizer/issues/163)
   ```pwsh
   $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
   setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
   ```

2. **Install NeuTTS**
   ```bash
   pip install neutts
   ```

   Or for a local editable install, clone this repository and run in the base folder:
   ```bash
   pip install -e .
   ```

   Alternatively to install all dependencies, including `onnxruntime` and `llama-cpp-python` (equivalent to steps 3 and 4 below):

   ```bash
   pip install neutts[all]
   ```

   or for an editable install:

   ```bash
   pip install -e .[all]
   ```

3. **(Optional) Install `llama-cpp-python` to use `.gguf` models.**

   ```bash
   pip install "neutts[llama]"
   ```

   Note that this installs `llama-cpp-python` without GPU support. To install with GPU support (e.g., CUDA, MPS) please refer to:
   https://pypi.org/project/llama-cpp-python/

4. **(Optional) Install `onnxruntime` to use the `.onnx` decoder.**
   ```bash
   pip install "neutts[onnx]"
   ```

## Examples

To get started with the example scripts, clone this repository and navigate into the project directory:

   ```bash
   git clone https://github.com/neuphonic/neutts.git
   cd neutts
   ```

Several examples are available, including a Jupyter notebook in the `examples` folder.

### Basic Example
Run the basic example script to synthesize speech:

```bash
python -m examples.basic_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_audio samples/jo.wav \
  --ref_text samples/jo.txt
```

To specify a particular model repo for the backbone or codec, add the `--backbone` argument. Available backbones are listed in [NeuTTS-Air](https://huggingface.co/collections/neuphonic/neutts-air) and [NeuTTS-Nano Multilingual Collection](https://huggingface.co/collections/neuphonic/neutts-nano-multilingual-collection) huggingface collections. 

> [!CAUTION]
> If you are using a non-English backbone, it is highly recommended to use a same-language reference for best performance. See the 'example reference files' section below to select an appropriate example reference.

### One-Code Block Usage

```python
from neutts import NeuTTS
import soundfile as sf

tts = NeuTTS(
   backbone_repo="neuphonic/neutts-nano", # or 'neuphonic/neutts-nano-q4-gguf' with llama-cpp-python installed
   backbone_device="cpu",
   codec_repo="neuphonic/neucodec",
   codec_device="cpu"
)
input_text = "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all."

ref_text = "samples/jo.txt"
ref_audio_path = "samples/jo.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
```

### Streaming

Speech can also be synthesised in _streaming mode_, where audio is generated in chunks and plays as generated. Note that this requires pyaudio to be installed. To do this, run:

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_codes samples/jo.pt \
  --ref_text samples/jo.txt
```

Again, a particular model repo can be specified with the `--backbone` argument - note that for streaming the model must be in GGUF format.

## Preparing References for Cloning

NeuTTS requires two inputs:

1. A reference audio sample (`.wav` file)
2. A text string

The model then synthesises the text as speech in the style of the reference audio. This is what enables NeuTTS models' instant voice cloning capability.

### Example Reference Files

You can find some ready-to-use references in the `samples` folder:

- English:
   - `dave.wav`
   - `jo.wav`
- Spanish:
   - `mateo.wav`
- German:
   - `greta.wav`
- French:
   - `juliette.wav`

### Guidelines for Best Results

For optimal performance, reference audio samples should be:

1. **Mono channel**
2. **16-44 kHz sample rate**
3. **3–15 seconds in length**
4. **Saved as a `.wav` file**
5. **Clean** — minimal to no background noise
6. **Natural, continuous speech** — like a monologue or conversation, with few pauses, so the model can capture tone effectively

## Guidelines for minimizing Latency

For optimal performance on-device:

1. Use the GGUF model backbones
2. Pre-encode references (see `examples/encode_reference.py` or `examples/basic_example.py`)
3. Use the [onnx codec decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder)

Take a look at this example in the [examples README](examples/README.md###minimal-latency-example) to get started.

## Responsibility

Every audio file generated by NeuTTS includes [Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth).

## Disclaimer

Don't use this model to do bad things… please.

## Developer Requirements

To run the pre commit hooks to contribute to this project run:

```bash
pip install pre-commit
```

Then:

```bash
pre-commit install
```

## Running Tests

First, install the dev requirements:

```
pip install -r requirements-dev.txt
```

To run the tests:

```
pytest tests/
```

To test loading of all the official backbone and codecs, use:

```
RUN_SLOW=true pytest tests/
```