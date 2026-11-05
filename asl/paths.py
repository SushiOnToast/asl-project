from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"

ARTIFACTS.mkdir(exist_ok=True)

LANDMARKER_TASK = MODELS / "hand_landmarker.task"
RAW_LANDMARKS = ARTIFACTS / "landmarks_raw.npz"
CLEAN_FEATURES = ARTIFACTS / "features_clean.npz"
SPLITS = ARTIFACTS / "splits.npz"
MODEL = ARTIFACTS / "mlp_best.pt"