import { NeuCodecWebDecoder } from "./decoder";
import { clampNormalize, pcmToWavBlob, playInAudioElement } from "./audio";

function qs<T extends HTMLElement>(sel: string): T {
  const el = document.querySelector(sel) as T | null;
  if (!el) throw new Error(`Missing element ${sel}`);
  return el;
}

async function readTextFile(file: File): Promise<string> {
  return await file.text();
}

async function readJsonArray(file: File): Promise<number[]> {
  const text = await file.text();
  const arr = JSON.parse(text);
  if (!Array.isArray(arr)) throw new Error("JSON is not an array");
  return arr.map((v) => Number(v));
}

async function fetchSample(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return await res.text();
}

async function generateCodesViaServer(
  serverUrl: string,
  inputText: string,
  refText: string,
  refCodes: number[]
): Promise<number[]> {
  const res = await fetch(serverUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_text: inputText,
      ref_text: refText,
      ref_codes: refCodes,
    }),
  });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  const data = await res.json();
  if (Array.isArray(data.ids)) return data.ids as number[];
  throw new Error("Unexpected server response");
}

async function main() {
  const inputTextEl = qs<HTMLInputElement>("#inputText");
  const refTextFileEl = qs<HTMLInputElement>("#refTextFile");
  const refCodesFileEl = qs<HTMLInputElement>("#refCodesFile");
  const useServerEl = qs<HTMLInputElement>("#useServer");
  const serverUrlEl = qs<HTMLInputElement>("#serverUrl");
  const modelPathEl = qs<HTMLInputElement>("#modelPath");
  const btnGenerate = qs<HTMLButtonElement>("#btnGenerate");
  const statusEl = qs<HTMLSpanElement>("#status");
  const playerEl = qs<HTMLAudioElement>("#player");

  const setStatus = (msg: string) => (statusEl.textContent = msg);

  btnGenerate.addEventListener("click", async () => {
    try {
      setStatus("Preparing...");

      const inputText = inputTextEl.value?.trim();
      if (!inputText) throw new Error("Please enter input text");

      // Load reference text
      let refText = "";
      if (refTextFileEl.files && refTextFileEl.files[0]) {
        refText = await readTextFile(refTextFileEl.files[0]);
      } else {
        refText = await fetchSample("/samples/ref_text.txt");
      }

      // Load reference codes
      let refCodes: number[] = [];
      if (refCodesFileEl.files && refCodesFileEl.files[0]) {
        refCodes = await readJsonArray(refCodesFileEl.files[0]);
      } else {
        const text = await fetchSample("/samples/ref_codes.json");
        refCodes = JSON.parse(text);
        if (!Array.isArray(refCodes))
          throw new Error("Sample ref_codes.json is invalid");
      }

      // Generate or use provided codes
      let ids: number[];
      if (useServerEl.checked) {
        const serverUrl =
          serverUrlEl.value?.trim() || "http://localhost:4000/api/generate";
        setStatus("Requesting token IDs from server...");
        ids = await generateCodesViaServer(
          serverUrl,
          inputText,
          refText,
          refCodes
        );
      } else {
        ids = refCodes;
      }

      if (!ids.length) throw new Error("No codes to decode");

      setStatus("Loading ONNX decoder...");
      const decoder = new NeuCodecWebDecoder({
        modelUrl: modelPathEl.value || "/models/neucodec_decoder.onnx.wasm",
      });
      await decoder.init();

      setStatus("Decoding...");
      const pcm = await decoder.decode(Int32Array.from(ids));
      const pcmNorm = clampNormalize(pcm);
      const wav = pcmToWavBlob(pcmNorm, 24000);
      await playInAudioElement(wav, playerEl);
      setStatus("Done");
    } catch (err: any) {
      console.error(err);
      setStatus(err?.message ?? String(err));
    }
  });
}

main().catch((e) => console.error(e));
