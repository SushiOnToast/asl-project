// returns the result of (features - mean) / std

import type { ModelMetadata } from "./types";

export function standardiseFeatures(
  features: Float32Array,
  metadata: Pick<ModelMetadata, "mean" | "std">,
): Float32Array {
  const { mean, std } = metadata;

  if (features.length !== 63) {
    throw new Error(`Expected 63 features, received ${features.length}`);
  }

  if (mean.length !== features.length || std.length !== features.length) {
    throw new Error("Metadata dimensions do not match feature dimensions");
  }

  const standardised = new Float32Array(features.length);

  for (let i = 0; i < features.length; i++) {
    if (!Number.isFinite(mean[i]) || !Number.isFinite(std[i]) || std[i] <= 0) {
      throw new Error(`Invalid standardisation metadata at index ${i}`);
    }

    standardised[i] = (features[i] - mean[i]) / std[i];
  }

  return standardised;
}
