# ASL Alphabet Fingerspeller

A model trained on ASL alphabet datasets that types letters as you sign them (fingerspelling), running live from your webcam. Everything runs on your own device, so no video is ever uploaded. See [Model](#model) for how it works.

## How to Use (Web App)

- A demo is available [here](https://sushiontoast.github.io/asl-project/).
- Click **Start camera**, hold up a sign from the ASL alphabet (A to Z), and keep it steady for about a second. The letter is typed into the sentence box.
- Use the **Space**, **Backspace** and **Clear** buttons (or the Space, Backspace and C keys) to edit the sentence.
- To type a double letter, drop your hand for a moment and sign the letter again.
- For best results: good lighting, a plain background, and your whole hand in view.
- To run it yourself, see [Running Locally](#running-locally).

![demo image](./demo.gif)

## Model

### How it works

Instead of looking at raw pixels, the model looks at the **shape** of your hand:

```
webcam frame -> MediaPipe hand landmarks (21 points x 3) -> normalise -> small neural network -> letter
```

1. **Find the hand.** MediaPipe's Hand Landmarker returns 21 key points (joints and fingertips) per frame.
2. **Normalise.** The points are made relative to the wrist and divided by hand size, so where your hand is in the frame and how far it is from the camera no longer matter. The y-axis is also rescaled by `height / width`, so a 640x480 webcam frame matches the square training images. Hands that fail basic checks (tiny or implausible detections) are rejected.
3. **Classify.** A small multilayer perceptron (MLP) predicts one of 26 letters from the 63 numbers.
4. **Smooth and commit.** Predictions are averaged over the last 8 frames. A letter is typed once it has stayed the confident top prediction (at least 70%) for about 20 frames.

The model was trained in Python with PyTorch, then exported to ONNX so the same network runs in the browser with ONNX Runtime Web. Both versions use identical preprocessing, and the web build is tested against reference outputs produced by the Python pipeline. See [Running Locally](#running-locally) to reproduce the training.

### Datasets Used

- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle, `grassknoted`): 3,000 images per letter, one signer, uniform backgrounds. Only the letters A to Z are used.
- [ASL-HG: American Sign Language Hand Gesture Image Dataset](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (Pranto et al., Mendeley Data, DOI 10.17632/j4y5w2c8w9.1, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): 10 volunteers with phone cameras, indoors and outdoors. Only the letters A to Z are used, not the digits.

Adding ASL-HG made the biggest difference: a model trained only on the first dataset scored about 96% on its own validation data but dropped to around 70 to 75% on new people. More signers is what helped, not more images from the same one.

<!-- TODO before publishing: confirm the licence of the Kaggle ASL Alphabet dataset and add attribution here if required. -->

I plan to use more data in the future to improve the model.

### Technologies Used

- **Python, PyTorch, scikit-learn, NumPy:** training and evaluation.
- **MediaPipe Hand Landmarker:** landmark detection in both the Python and web versions.
- **ONNX Runtime:** the exported model runs in the browser (ONNX Runtime Web) and in the desktop app.
- **TypeScript, Vite and Vitest:** the web app and its tests.
- **OpenCV and CustomTkinter:** the desktop app.

Python dependencies are listed in [python/requirements.txt](./python/requirements.txt) (training) and [python/requirements-app.txt](./python/requirements-app.txt) (desktop app).

### Model Architecture

```python
self.model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, n_classes),  # 26 letters
)
```

Training details:

- **Class weights** compensate for letters that MediaPipe detects less often (such as M and N).
- **Mirror augmentation** (negating x) is applied to training data only, so the model works with either hand.
- **Honest splits:** frames from the same recording are near-identical, so the first dataset is split into contiguous blocks, and ASL-HG is split by volunteer. The best checkpoint is chosen using volunteers the model has never seen.

### Results

Validation accuracy of the final model (the final held-out test set has not been reported yet):

| Validation set | Accuracy |
|---|---|
| ASL Alphabet (held-out frame blocks) | about 95% |
| ASL-HG (2 volunteers never seen in training) | about 91% |

The ASL-HG figure comes from only two people and varies between epochs, so treat it as approximate. Expect lower accuracy for people and cameras that differ a lot from the training data.

### Known Limitations

- **J and Z involve motion**, and a single frame can't capture it, so they are unreliable.
- **Similar-looking signs** such as M/N, R/U and D/O are confused more often than others. These depend on thumb position and finger crossing, which hand landmarks capture poorly.
- **Sign variants differ between people.** The two datasets don't always show a letter exactly the same way.
- **Landmark detection needs a clear view.** Poor lighting or a partly hidden hand can lose the hand entirely.

## Running Locally

The repository has two parts: `python/` (training, evaluation and a desktop app) and `web/` (the browser app). The exported model files in `web/public/model/` are committed, so you can run either app **without training anything**. Training outputs such as the extracted landmarks and checkpoints are not committed.

### Web app

Requires Node.js.

```bash
cd web
npm install
```

Copy MediaPipe's runtime files into `public/` once (PowerShell shown, use `cp -r` on macOS or Linux):

```powershell
New-Item -ItemType Directory -Force public\wasm
Copy-Item node_modules\@mediapipe\tasks-vision\wasm\* public\wasm\
```

Then:

```bash
npm run dev       # development server
npm run test      # unit tests, including parity checks against the Python pipeline
npm run build     # production build (set VITE_BASE=/your-repo/ when hosting in a subfolder)
```

Browsers only allow camera access on `localhost` or HTTPS.

### Desktop app (Python)

Requires Python 3.12. The desktop app uses the same ONNX model as the web app and does not need PyTorch.

```bash
cd python
python -m venv .venv-app
.venv-app\Scripts\activate          # macOS/Linux: source .venv-app/bin/activate
pip install -r requirements-app.txt
python -m app.main
```

To build a standalone Windows executable with PyInstaller, see the packaging command in [python/README.md](./python/README.md) <!-- TODO: add this file, or inline the command here -->. Windows may show a SmartScreen warning because the executable is unsigned.

### Reproducing the training pipeline

Requires Python 3.12, a [Kaggle API token](https://www.kaggle.com/docs/api) (saved to `~/.kaggle/access_token` or `kaggle.json`, never committed), and a few GB of disk space.

```bash
cd python
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The numbered scripts in `python/scripts/` are run in order:

| Stage | What it does | Output (in `python/artifacts/`) |
|---|---|---|
| Download | Fetches the MediaPipe hand model and the Kaggle dataset. ASL-HG is downloaded manually from Mendeley into `python/data/asl_hg/`. | `models/`, `data/` |
| Extract landmarks | Runs MediaPipe on every image (the first dataset takes about 40 minutes) | raw landmark files |
| Clean features | Normalises hands and drops unreliable samples | clean feature files |
| Merge and split | Combines both datasets and creates train, validation and test splits | splits file |
| Train | Trains the MLP with mirror augmentation | model checkpoint |
| Export | Writes the ONNX model, metadata and test fixtures into `web/public/model/` | web model files |

<!-- TODO: replace this table with the exact script names, or add a run_all script once it exists. -->

## Further Steps

This model and application are still in their early stages. Planned next steps:

- Train on more alphabet data from more signers.
- Improve the confusable letters, possibly with engineered features or a model that also looks at the hand image.
- Handle J and Z with a sequence model.
- Move on to word-level signs, using sequence models on landmark data.

## Credits

- Hand tracking: [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google.
- ASL-HG dataset: Pranto et al., Mendeley Data, DOI 10.17632/j4y5w2c8w9.1, CC BY 4.0.
- ASL Alphabet dataset: `grassknoted` on Kaggle.