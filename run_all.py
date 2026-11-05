"""
run_all.py: run the whole ASL pipeline, in order.

    python run_all.py              # run every step whose output is missing
    python run_all.py --force      # rerun everything from scratch
    python run_all.py --limit 50   # quick test: 50 images per class (implies --force)
    python run_all.py --skip-baseline
    python run_all.py --live       # launch the webcam app when training finishes

STEPS
    00 download hand model     -> models/hand_landmarker.task
    00 download dataset        -> data/            (needs Kaggle credentials)
    01 extract landmarks       -> artifacts/landmarks_raw.npz   (~40 min on full data)
    02 clean features          -> artifacts/features_clean.npz
    03 split data              -> artifacts/splits.npz
    04 baseline models         -> printed scores only
    05 train MLP               -> artifacts/mlp_best.pt

A step is skipped if its output file already exists (unless --force).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from asl import paths  # noqa: E402

# (label, script, output file or None = always run, extra-args hook)
STEPS = [
    ("Download hand model", "scripts/00_download_model.py", None),
    ("Download dataset", "scripts/00_download_dataset.py", None),
    ("Extract landmarks", "scripts/01_extract_landmarks.py", paths.RAW_LANDMARKS),
    ("Clean features", "scripts/02_clean_features.py", paths.CLEAN_FEATURES),
    ("Split data", "scripts/03_split_data.py", paths.SPLITS),
    ("Baseline models", "scripts/04_baseline.py", "baseline"),
    ("Train MLP", "scripts/05_train_mlp.py", paths.MODEL),
]


def run(script, extra_args=()):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}   # scripts can import `asl` without installing
    t = time.time()
    result = subprocess.run([sys.executable, str(ROOT / script), *extra_args], env=env, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nFAILED: {script} (exit code {result.returncode}). Fix it and rerun; "
                 f"finished steps will be skipped.")
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun every step")
    ap.add_argument("--limit", type=int, help="images per class for extraction (quick test)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--live", action="store_true", help="launch main.py at the end")
    args = ap.parse_args()

    force = args.force or args.limit is not None
    if args.limit:
        print(f"NOTE: --limit {args.limit}: artifacts will come from a SMALL SUBSET. "
              f"Run with --force and no --limit for the real model.\n")

    for i, (label, script, out) in enumerate(STEPS, 1):
        if out == "baseline":
            if args.skip_baseline:
                print(f"[{i}/{len(STEPS)}] {label}: skipped (--skip-baseline)")
                continue
            out = None
        elif out is not None and out.exists() and not force:
            print(f"[{i}/{len(STEPS)}] {label}: skipped ({out.name} exists)")
            continue

        print(f"\n[{i}/{len(STEPS)}] {label}  ({script})")
        extra = [str(args.limit)] if (args.limit and "extract" in script) else []
        secs = run(script, extra)
        print(f"[{i}/{len(STEPS)}] {label}: done in {secs:.0f}s")

    print("\nPipeline finished. Model saved to", paths.MODEL)
    if args.live:
        run("main.py")
    else:
        print("Run the webcam app with:  python main.py")


if __name__ == "__main__":
    main()