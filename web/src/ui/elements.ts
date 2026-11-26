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
  landmarksBtn: el<HTMLButtonElement>("landmarks-toggle"),
  tutorialBtn: el<HTMLButtonElement>("tutorial-toggle"),
  appView: el<HTMLDivElement>("app-view"),
  tutorialView: el<HTMLElement>("tutorial-view"),

  tutorialImage: el<HTMLImageElement>("tutorial-image"),
  tutorialTitle: el<HTMLHeadingElement>("tutorial-title"),
  tutorialDescription: el<HTMLParagraphElement>("tutorial-description"),
  tutorialCounter: el<HTMLParagraphElement>("tutorial-counter"),
  tutorialProgress: el<HTMLDivElement>("tutorial-progress"),
  tutorialBack: el<HTMLButtonElement>("tutorial-back"),
  tutorialNext: el<HTMLButtonElement>("tutorial-next"),
};
