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
const cameraBtn = el<HTMLButtonElement>("camera-toggle");

const ctx = canvas.getContext("2d");
if (!ctx) throw new Error("Canvas 2D context is not available");

let sentence = "";

function renderSentence(): void {
  sentenceEl.textContent = sentence + "▌";
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

let models: Models | null = null;
let cameraOn = false;
let cameraStream: MediaStream | null = null;
let stopDetection: (() => void) | null = null;

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
  const cameraStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = cameraStream;
  video.muted = true;
  await video.play();
}

function stopCamera(): void {
  if (stopDetection) {
    stopDetection();
    stopDetection = null;
  }

  // stop camera tracks
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.pause();
  video.srcObject = null;

  ctx!.clearRect(0, 0, canvas.width, canvas.height);

  cameraOn = false;
  cameraBtn.textContent = "Turn Camera On";
  letterEl.textContent = "–";
  letterEl.classList.remove("sure");
  barFill.style.width = "0%";

  setStatus("Camera off");
}

// same main function as main.py
function runLoop({ classifier, metadata, landmarker }: Models): () => void {
  const buffer = new ProbabilityBuffer(SMOOTH_FRAMES);
  const speller = new FingerSpeller(CONFIDENCE_THRESHOLD, HOLD_FRAMES);
  const drawing = new DrawingUtils(ctx!);
  let lastVideoTime = -1;
  let busy = false;
  let stopped = false;
  let frameId = 0;

  // to run every frame (basically while loop from main.py)
  async function tick(): Promise<void> {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      document.documentElement.style.setProperty("--ar", String(w / h));
    }

    const result = landmarker.detectForVideo(video, performance.now());
    ctx!.clearRect(0, 0, w, h);

    let letter: string | null = null;
    let confidence = 0;

    const landmarks = result.landmarks[0];
    if (landmarks) {
      // drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      //   color: "#ffffff",
      //   lineWidth: 2,
      // });
      // drawing.drawLandmarks(landmarks, { color: "#4ade80", radius: 3 });

      const probs = await classifyHand(landmarks, w, h, classifier, metadata);

      if (stopped) return;

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
    if (stopped) return;

    frameId = requestAnimationFrame(loop);
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

  frameId = requestAnimationFrame(loop);

  return () => {
    stopped = true;
    cancelAnimationFrame(frameId);
  };
}

async function main(): Promise<void> {
  renderSentence();
  let models: Models;
  try {
    models = await loadModels();
  } catch (err) {
    console.error(err);
    setStatus("Could not load model");
    return;
  }

  setStatus("Ready, start the camera");
  cameraBtn.disabled = false;

  cameraBtn.addEventListener("click", async () => {
    if (cameraOn) {
      stopCamera();
      return;
    }

    cameraBtn.disabled = true;
    setStatus("Waiting for camera permission...");

    try {
      await startCamera();

      cameraOn = true;
      cameraBtn.textContent = "Turn Camera Off";

      setStatus("No hand detected");

      stopDetection = runLoop(models!);
    } catch (err) {
      console.error(err);

      // Clean up in case the stream started but video.play() failed
      cameraStream?.getTracks().forEach((track) => track.stop());
      cameraStream = null;
      video.srcObject = null;

      const denied =
        err instanceof DOMException && err.name === "NotAllowedError";

      setStatus(
        denied
          ? "Camera permission denied. Allow it and try again."
          : "Could not start the camera.",
      );
    } finally {
      cameraBtn.disabled = false;
    }
  });
}

void main();
