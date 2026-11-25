import type { ASLClassifier } from "./classifier";
import { isValidHand, normaliseHand } from "./normalisation";
import { standardiseFeatures } from "./standardisation";
import type { Landmark, ModelMetadata, Prediction } from "./types";

// predict function from inference.py but condensed since functionality was delegated to other files
export async function classifyHand(
  landmarks: Landmark[],
  frameWidth: number,
  frameHeight: number,
  classifier: ASLClassifier,
  metadata: ModelMetadata,
): Promise<Float32Array | null> {
  const hand = normaliseHand(landmarks, frameWidth, frameHeight);
  if (!isValidHand(hand)) return null;
  const input = standardiseFeatures(hand.features, metadata);
  return classifier.predict(input);
}

// creating argmax and max functionality to return final prediction
export function topPrediction(
  probs: Float32Array | null,
  classes: string[],
): Prediction {
  if (!probs) return { letter: null, confidence: 0, valid: false };
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }
  return {
    letter: classes[best] ?? null,
    confidence: probs[best],
    valid: true,
  };
}
