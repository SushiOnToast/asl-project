"""
Utility data related to managing and accessing data imported from kaggle in the 
data/ directory
"""

import random
from pathlib import Path

import cv2

DATA_ROOT = Path("data")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_train_dir(root):
    """Return the folder that directly contains the class folders (A, B, ...)."""
    for p in root.rglob("A"): # recursive search
        if p.is_dir() and (p.parent / "B").is_dir():
            return p.parent
    return None


def list_images(class_dir):
    return sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS)


def main():
    train_dir = find_train_dir(DATA_ROOT)
    if train_dir is None:
        raise SystemExit(f"Couldn't find class folders under {DATA_ROOT}")
    print("Training folder:", train_dir)

    # image count
    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    total = 0
    for c in class_dirs:
        n = len(list_images(c))
        total += n
        print(f"{c.name:>8}: {n}")
    print(f"{len(class_dirs)} classes, {total} images total")

    # img properties
    files = list_images(train_dir / "A")
    img = cv2.imread(str(files[0]))
    if img is None:
        raise SystemExit(f"Could not read {files[0]}")
    print(f"Shape: {img.shape}, dtype: {img.dtype}")

    # 3 random image sample
    try:
        for f in random.sample(files, 3):
            img = cv2.imread(str(f))
            if img is None:
                continue
            cv2.imshow("Random A (any key = next, q = quit)", img)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()