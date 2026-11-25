import { describe, expect, it } from "vitest";
import { FingerSpeller, ProbabilityBuffer } from "./speller";

type Frame = [string | null, number];
// create a representation of a series of frames with letter predictions and such to test the fingerspeller
const rep = (letter: string | null, conf: number, n: number): Frame[] =>
  Array.from({ length: n }, () => [letter, conf] as Frame);

// runs through frames and returns fingerspeller result after each
function run(frames: Frame[]): string[] {
  const speller = new FingerSpeller();
  const typed: string[] = [];
  for (const [letter, conf] of frames) {
    const out = speller.update(letter, conf);
    if (out) typed.push(out);
  }
  return typed;
}

// run tests
describe("FingerSpeller", () => {
  it("commits a held letter exactly once", () => {
    expect(run(rep("B", 0.9, 25))).toEqual(["B"]);
  });

  it("ignores a sign that is not held long enough", () => {
    expect(run([...rep("B", 0.9, 10), ...rep("L", 0.9, 25)])).toEqual(["L"]);
  });

  it("allows a repeat after the hold is broken", () => {
    expect(
      run([...rep("B", 0.9, 25), ...rep(null, 0, 3), ...rep("B", 0.9, 25)]),
    ).toEqual(["B", "B"]);
  });

  it("types nothing below the confidence threshold", () => {
    expect(run(rep("B", 0.5, 40))).toEqual([]);
  });

  it("reports progress", () => {
    const s = new FingerSpeller(0.7, 20);
    for (let i = 0; i < 10; i++) s.update("A", 0.9);
    expect(s.progress).toBeCloseTo(0.5);
  });
});

// test buffer
describe("ProbabilityBugger", () => {
  it("averages and keeps only the latest items", () => {
    const b = new ProbabilityBuffer(2);
    expect(b.mean()).toBeNull;
    b.push(Float32Array.from([1, 0]));
    b.push(Float32Array.from([0, 1]));
    b.push(Float32Array.from([0, 1])); // should drop the first push after this one

    const m = b.mean()!;
    expect(m[0]).toBeCloseTo(0);
    expect(m[1]).toBeCloseTo(1);
  });
});
