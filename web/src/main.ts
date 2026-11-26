import "./style.css";

import { elements } from "./ui/elements";
import { setupTutorial } from "./ui/tutorial";
import { createSentenceEditor } from "./ui/sentence";
import { setupKeyboard } from "./ui/keyboard";
import { updatePrediction, resetPrediction, setStatus } from "./ui/prediction";

import { loadModels } from "./asl/runtime/loadModels";
import { startCamera, stopCamera } from "./asl/runtime/camera";
import { startDetectionLoop } from "./asl/runtime/detectionLoop";

import type { ASLModels } from "./asl/runtime/types";

// Application state
let models: ASLModels | null = null;
let cameraStream: MediaStream | null = null;
let stopDetection: (() => void) | null = null;
let cameraOn = false;
let starting = false;
let showLandmarks = true;

const sentence = createSentenceEditor();

// Keyboard and sentence controls
setupKeyboard({
  addText: sentence.addText,
  backspace: sentence.backspace,
  clear: sentence.clear,
});

// Camera cleanup
function turnCameraOff(): void {
  stopDetection?.();
  stopDetection = null;

  if (cameraStream) {
    stopCamera(cameraStream, elements.video, elements.canvas);
  }

  cameraStream = null;
  cameraOn = false;

  elements.cameraBtn.textContent = "Start Camera";

  resetPrediction();
  setStatus("Camera off");
}

// Camera startup
async function turnCameraOn(): Promise<void> {
  if (!models || starting) return;

  starting = true;
  elements.cameraBtn.disabled = true;

  setStatus("Waiting for camera permission...");

  try {
    cameraStream = await startCamera(elements.video);

    cameraOn = true;
    elements.cameraBtn.textContent = "Stop Camera";

    setStatus("No hand detected");

    stopDetection = startDetectionLoop({
      models,
      video: elements.video,
      canvas: elements.canvas,
      showLandmarks: () => showLandmarks,
      onUpdate: (update) => {
        updatePrediction(update);
        setStatus(update.status);
      },

      onTypedLetter: (letter) => {
        sentence.addText(letter);
      },
    });
  } catch (error) {
    console.error("Camera error:", error);

    if (cameraStream) {
      stopCamera(cameraStream, elements.video, elements.canvas);

      cameraStream = null;
    }

    const denied =
      error instanceof DOMException && error.name === "NotAllowedError";

    setStatus(
      denied
        ? "Camera permission denied. Allow it and try again."
        : "Could not start the camera.",
    );
  } finally {
    starting = false;
    elements.cameraBtn.disabled = false;
  }
}

// Main camera button
function setupCameraButton(): void {
  elements.cameraBtn.addEventListener("click", async () => {
    if (cameraOn) {
      turnCameraOff();
    } else {
      await turnCameraOn();
    }

    elements.cameraBtn.blur();
  });
}

// Landmark visibility toggle
function setupLandmarksButton(): void {
  elements.landmarksBtn.addEventListener("click", () => {
    showLandmarks = !showLandmarks;

    elements.landmarksBtn.textContent = showLandmarks
      ? "Hide Skeleton"
      : "Show Skeleton";

    elements.landmarksBtn.setAttribute(
      "aria-pressed",
      String(showLandmarks),
    );

    elements.landmarksBtn.blur();
  });
}

// Application entry point
async function main(): Promise<void> {
  try {
    setStatus("Loading model...");

    models = await loadModels();

    setStatus("Ready, start the camera");

    elements.cameraBtn.disabled = false;
    setupCameraButton();
    setupLandmarksButton();
  } catch (error) {
    console.error("Model loading error:", error);
    setStatus("Could not load model");
  }
}

// Cleanup when leaving the page
window.addEventListener("pagehide", () => {
  if (cameraStream) {
    turnCameraOff();
  }
});

setupTutorial();

void main();
