import type { Landmark, NormalisedHand } from "./types";

const EPS = 1e-6;

export function normaliseHand(
  landmarks: Landmark[],
  frameWidth: number,
  frameHeight: number,
): NormalisedHand {
  if (landmarks.length != 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks.length}`);
  }

  if (frameWidth <= 0 || frameHeight <= 0) {
    throw new Error("Frame dimensions must be positive");
  }

  const hand = landmarks.map(({ x, y, z }) => [
    x,
    y * (frameHeight / frameWidth),
    z,
  ]);

  // make landmarks relative to wrist
  const wrist = hand[0];

  const relative = hand.map(([x, y, z]) => [
    x - wrist[0],
    y - wrist[1],
    z - wrist[2],
  ]);

  // calculating hand size
  const handSize = Math.max(Math.hypot(relative[9][0], relative[9][1]), EPS);

  // scale and flatten into 63 features.
  const features = new Float32Array(
    relative.flatMap(([x, y, z]) => [x / handSize, y / handSize, z / handSize]),
  );

  return {
    features,
    size: handSize,
  };
}

// function to check whether input hand valid for inference
export function isValidHand(hand: NormalisedHand): boolean {
  if (hand.size < 0.1) {
    return false;
  }

  for (const feature of hand.features) {
    if (Math.abs(feature) > 5) {
      return false;
    }
  }

  return true;
}
