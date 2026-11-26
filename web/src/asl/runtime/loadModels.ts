import * as ort from "onnxruntime-web";

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { ASLClassifier } from "../classifier";
import type { ModelMetadata } from "../types";
import type { ASLModels } from "./types";

import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";

const BASE = import.meta.env.BASE_URL;

export async function loadModels(): Promise<ASLModels> {
  ort.env.wasm.wasmPaths = {
    wasm: wasmUrl,
    mjs: mjsUrl,
  };

  const response = await fetch(`${BASE}model/metadata.json`);

  if (!response.ok) {
    throw new Error(`Could not load metadata.json (${response.status})`);
  }

  const metadata = (await response.json()) as ModelMetadata;

  const classifier = await ASLClassifier.create(
    `${BASE}model/asl_mlp.onnx`,
    metadata,
  );

  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);

  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: `${BASE}model/hand_landmarker.task`,
    },
    runningMode: "VIDEO",
    numHands: 1,
  });

  return { classifier, metadata, landmarker };
}
