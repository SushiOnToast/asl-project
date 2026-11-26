function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);

  if (!node) {
    throw new Error(`Missing element #${id}`);
  }

  return node as T;
}

export const elements = {
  video: el<HTMLVideoElement>("video"),
  canvas: el<HTMLCanvasElement>("overlay"),
  status: el<HTMLDivElement>("status"),
  letter: el<HTMLDivElement>("letter"),
  barFill: el<HTMLDivElement>("bar-fill"),
  sentence: el<HTMLDivElement>("sentence"),
  cameraBtn: el<HTMLButtonElement>("camera-toggle"),
};
