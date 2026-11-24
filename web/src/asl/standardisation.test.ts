import { describe, it, expect } from "vitest";

import fixtures from "../../public/model/fixtures.json";
import metadata from "../../public/model/metadata.json";

import { normaliseHand } from "./normalisation";
import { standardiseFeatures } from "./standardisation";

import type { Landmark } from "./types";

describe("Python / Typescript standardisation parity", () => {
  it.each(fixtures)("matches Python standardisation", (fixture) => {
    const landmarks: Landmark[] = fixture.landmarks.map(([x, y, z]) => ({
      x,
      y,
      z,
    }));

    const normalised = normaliseHand(landmarks, fixture.width, fixture.height);

    const result = standardiseFeatures(normalised.features, metadata);

    expect(result.length).toBe(63);

    for (let i = 0; i < 63; i++) {
      expect(result[i]).toBeCloseTo(fixture.standardised_features[i], 4);
    }
  });
});
