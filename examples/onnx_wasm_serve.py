from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
from neuttsair.neutts import NeuTTSAir


class GenerateRequest(BaseModel):
    input_text: str
    ref_text: str
    ref_codes: list[int]
    backbone: str | None = None


app = FastAPI(title="NeuTTSAir WASM Server Stub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create a single TTS instance for reuse
_tts: NeuTTSAir | None = None


def _get_tts(backbone: str | None) -> NeuTTSAir:
    global _tts
    if _tts is None or (backbone and backbone != "neuphonic/neutts-air"):
        _tts = NeuTTSAir(
            backbone_repo=backbone or "neuphonic/neutts-air",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec-onnx-decoder",
            codec_device="cpu",
        )
    return _tts


@app.post("/api/generate")
def generate(req: GenerateRequest):
    tts = _get_tts(req.backbone)

    ref_codes = req.ref_codes

    prompt_ids = tts._apply_chat_template(ref_codes, req.ref_text, req.input_text)

    output_str = tts._infer_torch(prompt_ids)

    # Extract speech token IDs
    ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", output_str)]

    return {"ids": ids, "codes_str": output_str}
