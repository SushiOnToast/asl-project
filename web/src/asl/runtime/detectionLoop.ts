import { classifyHand, topPrediction } from "../predict";
import { FingerSpeller, ProbabilityBuffer } from "../speller";

import type { ASLModels, DetectionUpdate } from "./types";

interface DetectionLoopOptions {
  models: ASLModels;
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  onUpdate: (update: DetectionUpdate) => void;
  onTypedLetter: (letter: string) => void;
}

const CONFIDENCE_THRESHOLD = 0.8;
const HOLD_FRAMES = 10;
const SMOOTH_FRAMES = 8;

export function startDetectionLoop({
  models,
  video,
  canvas,
  onUpdate,
  onTypedLetter,
}: DetectionLoopOptions): () => void {
  const { classifier, metadata, landmarker } = models;

  const buffer = new ProbabilityBuffer(SMOOTH_FRAMES);
  const speller = new FingerSpeller(CONFIDENCE_THRESHOLD, HOLD_FRAMES);

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Canvas 2D context is not available");
  }

  const ctx: CanvasRenderingContext2D = context;

  let lastVideoTime = -1;
  let busy = false;
  let stopped = false;
  let frameId = 0;

  async function tick(): Promise<void> {
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;

      document.documentElement.style.setProperty(
        "--ar",
        String(width / height),
      );
    }

    const result = landmarker.detectForVideo(video, performance.now());

    ctx.clearRect(0, 0, width, height);

    let letter: string | null = null;
    let confidence = 0;
    let status = "No hand detected";

    const landmarks = result.landmarks[0];

    if (landmarks) {
      const probabilities = await classifyHand(
        landmarks,
        width,
        height,
        classifier,
        metadata,
      );

      if (stopped) return;

      if (probabilities) {
        buffer.push(probabilities);

        const prediction = topPrediction(buffer.mean(), metadata.classes);

        letter = prediction.letter;
        confidence = prediction.confidence;

        status =
          confidence >= CONFIDENCE_THRESHOLD ? "Hold steady…" : "Not sure yet";
      } else {
        buffer.clear();
        status = "Hand too small or unclear, move closer";
      }
    } else {
      buffer.clear();
    }

    const typed = speller.update(letter, confidence);

    if (typed) {
      onTypedLetter(typed);
    }

    onUpdate({
      letter,
      confidence,
      progress: speller.progress,
      status,
    });
  }

  function loop(): void {
    if (stopped) return;

    frameId = requestAnimationFrame(loop);

    if (busy || video.readyState < 2 || video.currentTime === lastVideoTime) {
      return;
    }

    lastVideoTime = video.currentTime;
    busy = true;

    tick()
      .catch((error) => console.error("Detection error:", error))
      .finally(() => {
        busy = false;
      });
  }

  frameId = requestAnimationFrame(loop);

  return () => {
    stopped = true;
    cancelAnimationFrame(frameId);
  };
}
