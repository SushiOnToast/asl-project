import { describe, it, expect } from "vitest";
import { normaliseHand, isValidHand } from "./normalisation";
import type { Landmark } from "./types";
import fixtures from "../../public/model/fixtures.json";

describe("normaliseHand", () => {
  const landmarks: Landmark[] = Array.from({ length: 21 }, (_, i) => ({
    x: i * 0.01,
    y: i * 0.015,
    z: i * 0.005,
  }));

  it("returns exactly 63 features", () => {
    const result = normaliseHand(landmarks, 640, 480);

    expect(result.features.length).toBe(63);
  });

  it("makes the wrist coordinates zero", () => {
    const result = normaliseHand(landmarks, 640, 480);

    expect(result.features[0]).toBeCloseTo(0);
    expect(result.features[1]).toBeCloseTo(0);
    expect(result.features[2]).toBeCloseTo(0);
  });

  it("produces a valid hand for suitable landmarks", () => {
    const result = normaliseHand(landmarks, 640, 480);

    expect(isValidHand(result)).toBe(true);
  });

  it("rejects hands that are too small", () => {
    const tinyHand = landmarks.map((point) => ({
      x: point.x * 0.001,
      y: point.y * 0.001,
      z: point.z * 0.001,
    }));

    const result = normaliseHand(tinyHand, 640, 480);

    expect(isValidHand(result)).toBe(false);
  });
});

// testing with actual features from python folder now

describe("Python / Typescript preprocessing parity", () => {
  it.each(fixtures)(
    "matches Python preprocessing (valid: $valid)",
    (fixture) => {
      const landmarks: Landmark[] = fixture.landmarks.map(([x, y, z]) => ({
        x,
        y,
        z,
      }));

      const result = normaliseHand(landmarks, fixture.width, fixture.height);

      expect(result.features.length).toBe(63);

      // comparing features
      for (let i = 0; i < 63; i++) {
        expect(result.features[i]).toBeCloseTo(fixture.features[i], 4);
      }

      // compare hand size
      expect(result.size).toBeCloseTo(fixture.size, 5);

      // compare validity
      expect(isValidHand(result)).toBe(fixture.valid);
    },
  );
});
