import { elements } from "./elements";

export function createSentenceEditor() {
  let sentence = "";

  function render(): void {
    const cursor = document.createElement("span");
    cursor.textContent = "▌";
    cursor.className = "animate-pulse";

    elements.sentence.replaceChildren(
      document.createTextNode(sentence),
      cursor,
    );
  }

  function addText(text: string): void {
    sentence += text;
    render();
  }

  function backspace(): void {
    sentence = sentence.slice(0, -1);
    render();
  }

  function clear(): void {
    sentence = "";
    render();
  }

  render();

  return {
    addText,
    backspace,
    clear,
  };
}
