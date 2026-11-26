import { elements } from "./elements";

interface PredictionDisplay {
  letter: string | null;
  confidence: number;
  progress: number;
}

export function updatePrediction({
  letter,
  confidence,
  progress,
}: PredictionDisplay): void {
  if (letter) {
    elements.letter.innerHTML = `${letter}<small class="ml-2 text-sm font-normal">${Math.round(confidence * 100)}%</small>`;
  } else {
    elements.letter.textContent = "–";
  }

  elements.letter.classList.toggle(
    "text-accent",
    letter !== null && confidence >= 0.8,
  );

  elements.letter.classList.toggle(
    "text-muted",
    letter === null || confidence < 0.8,
  );

  elements.barFill.style.width = `${Math.round(progress * 100)}%`;
}

export function setStatus(status: string): void {
  elements.status.textContent = status;
}

export function resetPrediction(): void {
  elements.letter.textContent = "–";
  elements.letter.classList.remove("text-accent");
  elements.letter.classList.add("text-muted");
  elements.barFill.style.width = "0%";
}
