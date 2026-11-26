import type { ASLClassifier } from "../classifier";
import type { ModelMetadata } from "../types";
import type { HandLandmarker } from "@mediapipe/tasks-vision";

export interface ASLModels {
  classifier: ASLClassifier;
  metadata: ModelMetadata;
  landmarker: HandLandmarker;
}

export interface DetectionUpdate {
  letter: string | null;
  confidence: number;
  progress: number;
  status: string;
}
