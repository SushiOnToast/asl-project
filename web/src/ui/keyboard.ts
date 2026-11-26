import { elements } from "./elements";

interface KeyboardActions {
  addText: (text: string) => void;
  backspace: () => void;
  clear: () => void;
}

export function setupKeyboard(actions: KeyboardActions): void {
  elements.cameraBtn.blur();

  document.getElementById("space")?.addEventListener("click", () => {
    actions.addText(" ");
  });

  document.getElementById("backspace")?.addEventListener("click", () => {
    actions.backspace();
  });

  document.getElementById("clear")?.addEventListener("click", () => {
    actions.clear();
  });

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key === " ") {
      e.preventDefault();
      actions.addText(" ");
    } else if (e.key === "Backspace") {
      e.preventDefault();
      actions.backspace();
    } else if (e.key.toLowerCase() === "c") {
      actions.clear();
    }
  });
}
