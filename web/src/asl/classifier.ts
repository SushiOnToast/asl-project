import * as ort from "onnxruntime-web";
import type { ModelMetadata } from "./types";

ort.env.wasm.numThreads = 1; // disable multithreading for easier deployment

export class ASLClassifier {
  private session: ort.InferenceSession;
  private classes: string[];

  private constructor(session: ort.InferenceSession, classes: string[]) {
    this.session = session;
    this.classes = classes;
  }

  static async create(
    modelUrl: string,
    metadata: ModelMetadata,
  ): Promise<ASLClassifier> {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });

    return new ASLClassifier(session, metadata.classes);
  }

  softmax(values: Float32Array): Float32Array {
    const maxLogit = Math.max(...values);
    const exponentials = Array.from(values, (value) =>
      Math.exp(value - maxLogit),
    );

    const sum = exponentials.reduce((total, value) => total + value, 0);

    return Float32Array.from(exponentials.map((value) => value / sum));
  }

  async predict(features: Float32Array): Promise<Float32Array> {
    if (features.length !== 63) {
      throw new Error(`Expected 63 features, received ${features.length}`);
    }

    const input = new ort.Tensor("float32", features, [1, 63]);

    const outputs = await this.session.run({
      features: input,
    });

    const logits = outputs.logits;

    if (!logits) {
      throw new Error("onnx model did not return logits");
    }

    const values = logits.data as Float32Array;

    if (values.length !== this.classes.length) {
      throw new Error(
        `Expected ${this.classes.length} logits, received ${values.length}`,
      );
    }

    return this.softmax(values);
  }

  getClasses(): string[] {
    return [...this.classes];
  }
}
