
import { elements } from "./elements";
import { tutorialSlides } from "./tutorialSlides";

export function setupTutorial(): void {
  let currentSlide = 0;
  let isOpen = false;

  const {
    tutorialBtn: button,
    appView,
    tutorialView: view,
    tutorialImage: image,
    tutorialTitle: title,
    tutorialDescription: description,
    tutorialCounter: counter,
    tutorialProgress: progress,
    tutorialBack: back,
    tutorialNext: next,
  } = elements;

  function renderSlide(): void {
    const slide = tutorialSlides[currentSlide];

    image.src = slide.image;
    image.alt = slide.title;

    title.textContent = slide.title;
    description.textContent = slide.description;

    counter.textContent =
      `Slide ${currentSlide + 1} of ${tutorialSlides.length}`;

    back.disabled = currentSlide === 0;

    next.textContent =
      currentSlide === tutorialSlides.length - 1 ? "Finish" : "Next";

    // Update progress indicators
    progress.replaceChildren();

    tutorialSlides.forEach((_, index) => {
      const dot = document.createElement("div");

      dot.className =
        "h-2 w-6 rounded-full " +
        (index === currentSlide ? "bg-accent" : "bg-hover");

      progress.appendChild(dot);
    });
  }

  function openTutorial(): void {
    isOpen = true;
    currentSlide = 0;

    appView.hidden = true;
    view.hidden = false;

    button.textContent = "Close Tutorial";
    button.setAttribute("aria-expanded", "true");

    renderSlide();
  }

  function closeTutorial(): void {
    isOpen = false;

    view.hidden = true;
    appView.hidden = false;

    button.textContent = "How to use";
    button.setAttribute("aria-expanded", "false");
  }

  back.addEventListener("click", () => {
    if (currentSlide > 0) {
      currentSlide--;
      renderSlide();
    }
  });

  next.addEventListener("click", () => {
    if (currentSlide === tutorialSlides.length - 1) {
      closeTutorial();
    } else {
      currentSlide++;
      renderSlide();
    }
  });

  button.addEventListener("click", () => {
    if (isOpen) {
      closeTutorial();
    } else {
      openTutorial();
    }

    button.blur();
  });
}
