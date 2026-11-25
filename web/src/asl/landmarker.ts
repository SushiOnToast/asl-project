import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export async function createLandmarker(): Promise<HandLandmarker> {
  const base = import.meta.env.BASE_URL;
  const fileset = await FilesetResolver.forVisionTasks(`${base}wasm`);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${base}model/hand_landmarker.task` },
    runningMode: "VIDEO",
    numHands: 1,
  });
}
