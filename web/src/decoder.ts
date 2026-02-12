import * as ort from "onnxruntime-web";

export type DecoderOptions = {
  modelUrl: string;
};

export class NeuCodecWebDecoder {
  private session?: ort.InferenceSession;
  private inputName?: string;
  private outputName?: string;

  constructor(private readonly options: DecoderOptions) {}

  async init(): Promise<void> {
    ort.env.wasm.wasmPaths = "/dist/"; // need to be copied from onnxruntime-web
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency || 4);

    console.log("Initializing decoder with model URL:", this.options.modelUrl);

    this.session = await ort.InferenceSession.create(this.options.modelUrl);

    const anySession = this.session 
    this.inputName = anySession.inputNames?.[0];
    this.outputName = anySession.outputNames?.[0];
  }

  async decode(codes: Int32Array): Promise<Float32Array> {
    if (!this.session)
      throw new Error("Decoder session not initialized. Call init() first.");

    const inputTensor = new ort.Tensor("int32", codes, [1, 1, codes.length]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputName!] = inputTensor;

    const results = await this.session.run(feeds);

    const out = results[this.outputName!]?.data as Float32Array;
   
    return out;
  }
}
