interface TutorialSlide {
  image: string;
  title: string;
  description: string;
}
const BASE = import.meta.env.BASE_URL;

export const tutorialSlides: TutorialSlide[] = [
  {
    image: `${BASE}tutorial/slide_1.png`,
    title: "Allow camera access",
    description:
      "Click on 'Start Camera' and enable your camera so the app can detect your hand.",
  },
  {
    image: `${BASE}tutorial/slide_2.png`,
    title: "Position your hand",
    description:
      "Keep your hand visible within the camera frame. Make sure your fingers are clearly visible.",
  },
  {
    image: `${BASE}tutorial/slide_3.png`,
    title: "Hold your hand steady",
    description:
      "The app will predict the letter you're signing. Hold your handshape until the progress bar fills up.",
  },
  {
    image: `${BASE}tutorial/slide_4.png`,
    title: "Build your sentence",
    description:
      "Your recognised letters will appear in the sentence box. Use Space (Space Bar), Backspace (Backspace Key) and Clear (C key) to edit your text.",
  },
];