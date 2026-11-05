"""
extract the hand landmarks from the dataset images, producing features for training
"""

import sys

import cv2
import mediapipe as mp
import numpy as np

from asl.hand_landmarker import make_landmarker, landmarks_to_array
from asl.data import find_train_dir, list_images
from asl.paths import *

SKIP = {"space", "nothing", "del"}

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    train_dir = find_train_dir(DATA)
    if train_dir is None:
        raise SystemExit(f"Couldn't find class folders under {DATA}")

    class_dirs = sorted(p for p in train_dir.iterdir()
                        if p.is_dir() and p.name not in SKIP)

    X_list, labels, paths = [], [], []   # parallel lists

    with make_landmarker() as landmarker:
        for class_dir in class_dirs:
            files = list_images(class_dir)
            if limit:
                files = files[:limit]

            success = failed = 0
            for file in files:
                img = cv2.imread(str(file))
                if img is None:
                    failed += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    X_list.append(landmarks_to_array(result.hand_landmarks[0]))
                    labels.append(class_dir.name)
                    paths.append(str(file))
                    success += 1
                else:
                    failed += 1

            total = success + failed
            print(f"{class_dir.name:>3}: {success}/{total} detected "
                  f"({100 * failed / max(total, 1):.1f}% failed)")

    X = np.array(X_list, dtype=np.float32)
    np.savez_compressed(RAW_LANDMARKS, X=X, y=np.array(labels),
                        paths=np.array(paths))
    print(f"\nSaved landmarks_raw.npz: X shape = {X.shape}")


if __name__ == "__main__":
    main()