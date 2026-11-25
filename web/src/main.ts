import "./style.css";
import * as ort from "onnxruntime-web";
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import { ASLClassifier } from "./asl/classifier";
import { classifyHand, topPrediction } from "./asl/predict";
import { FingerSpeller, ProbabilityBuffer } from "./asl/speller";
import type { ModelMetadata } from "./asl/types";

import mjsUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import wasmUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url";

const BASE = import.meta.env.BASE_URL;
const CONFIDENCE_THRESHOLD = 0.8;
const HOLD_FRAMES = 20;
const SMOOTH_FRAMES = 8; // buffer smoothing
const SENTENCE_VISIBLE_CHARS = 40;

// helper to get element by id
function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element #${id}`);
  return node as T;
}

const video = el<HTMLVideoElement>("video");
const canvas = el<HTMLCanvasElement>("overlay");
const statusEl = el<HTMLDivElement>("status");
const letterEl = el<HTMLDivElement>("letter");
const barFill = el<HTMLDivElement>("bar-fill");
const sentenceEl = el<HTMLDivElement>("sentence");
const startBtn = el<HTMLButtonElement>("start");

const ctx = canvas.getContext("2d");
if (!ctx) throw new Error("Canvas 2D context is not available");

let sentence = "";

function renderSentence(): void {
  sentenceEl.textContent = sentence.slice(-SENTENCE_VISIBLE_CHARS) + "▌";
}

function addText(s: string): void {
  sentence += s;
  renderSentence();
}

function backspace(): void {
  sentence = sentence.slice(0, -1);
  renderSentence();
}

function clearText(): void {
  sentence = "";
  renderSentence();
}

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function wireButton(id: string, action: () => void): void {
  const btn = el<HTMLButtonElement>(id);
  btn.addEventListener("click", () => {
    action();
    btn.blur();
  });
}

wireButton("space", () => addText(" "));
wireButton("backspace", backspace);
wireButton("clear", clearText);

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === " ") {
    e.preventDefault();
    addText(" ");
  } else if (e.key === "Backspace") {
    e.preventDefault();
    backspace();
  } else if (e.key.toLowerCase() === "c") clearText();
});

interface Models {
  classifier: ASLClassifier;
  metadata: ModelMetadata;
  landmarker: HandLandmarker;
}

async function loadModels(): Promise<Models> {
  ort.env.wasm.wasmPaths = {
    wasm: wasmUrl,
    mjs: mjsUrl,
  };

  const res = await fetch(`${BASE}model/metadata.json`);
  if (!res.ok) throw new Error(`Could not load metadata.json (${res.status})`);
  const metadata = (await res.json()) as ModelMetadata;

  const classifier = await ASLClassifier.create(
    `${BASE}model/asl_mlp.onnx`,
    metadata,
  );

  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${BASE}model/hand_landmarker.task` },
    runningMode: "VIDEO",
    numHands: 1,
  });

  return { classifier, metadata, landmarker };
}

async function startCamera(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = stream;
  video.muted = true;
  await video.play();
}

// same main function as main.py
function runLoop({ classifier, metadata, landmarker }: Models): void {
  const buffer = new ProbabilityBuffer(SMOOTH_FRAMES);
  const speller = new FingerSpeller(CONFIDENCE_THRESHOLD, HOLD_FRAMES);
  const drawing = new DrawingUtils(ctx!);
  let lastVideoTime = -1;
  let busy = false;

  // to run every frame (basically while loop from main.py)
  async function tick(): Promise<void> {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const result = landmarker.detectForVideo(video, performance.now());
    ctx!.clearRect(0, 0, w, h);

    let letter: string | null = null;
    let confidence = 0;

    const landmarks = result.landmarks[0];
    if (landmarks) {
      drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: "#ffffff",
        lineWidth: 2,
      });
      drawing.drawLandmarks(landmarks, { color: "#4ade80", radius: 3 });

      const probs = await classifyHand(landmarks, w, h, classifier, metadata);
      if (probs) {
        buffer.push(probs);
        const top = topPrediction(buffer.mean(), metadata.classes);
        letter = top.letter;
        confidence = top.confidence;
        setStatus(
          confidence >= CONFIDENCE_THRESHOLD ? "Hold steady…" : "Not sure yet",
        );
      } else {
        buffer.clear();
        setStatus("Hand too small or unclear, move closer");
      }
    } else {
      buffer.clear();
      setStatus("No hand detected");
    }

    const typed = speller.update(letter, confidence);
    if (typed) addText(typed);

    if (letter) {
      letterEl.innerHTML = `${letter}<small>${Math.round(confidence * 100)}%</small>`;
    } else {
      letterEl.textContent = "–";
    }
    letterEl.classList.toggle(
      "sure",
      letter !== null && confidence >= CONFIDENCE_THRESHOLD,
    );
    barFill.style.width = `${Math.round(speller.progress * 100)}%`;
  }

  function loop(): void {
    requestAnimationFrame(loop);
    if (busy || video.readyState < 2 || video.currentTime === lastVideoTime)
      return;
    lastVideoTime = video.currentTime; // timestamps must increase and to not send the same frame we check that before
    busy = true;
    tick()
      .catch((err) => console.error(err))
      .finally(() => {
        busy = false; // an error must never freeze the loop
      });
  }

  requestAnimationFrame(loop);
}

async function main(): Promise<void> {
  renderSentence();
  let models: Models;
  try {
    models = await loadModels();
  } catch (err) {
    console.error(err);
    setStatus("Could not load model");
  }

  setStatus("Ready, start the camera");
  startBtn.disabled = false;

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    setStatus("Waiting for camera permission...");
    try {
      await startCamera();
    } catch (err) {
      console.error(err);
      const denied =
        err instanceof DOMException && err.name === "NotAllowedError";
      setStatus(
        denied
          ? "Camera permission denied. Allow it and reload."
          : "Could not start the camera.",
      );
      startBtn.disabled = false;
      return;
    }

    startBtn.style.display = "none";
    setStatus("No hand detected");
    runLoop(models);
  });
}

void main();
