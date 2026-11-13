"""
extract the hand landmarks from the dataset images, producing features for training
"""

import re
import sys
import time
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np

from asl.hand_landmarker import make_landmarker, landmarks_to_array
from asl.data import list_images
from asl import paths

SOURCE = "asl_hg"
SIGNER_RE = re.compile(r"P(\d+)_")


def signer_id(path):
    m = SIGNER_RE.match(path.name)
    return int(m.group(1)) if m else -1


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    root = paths.DATA_HG
    if not root.exists():
        raise SystemExit(f"Couldn't find class folders under {paths.DATA}")

    class_dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and len(p.name) == 1 and p.name.isalpha()
    )

    X_list, labels, img_paths, signers = [], [], [], []
    stats = defaultdict(
        lambda: [0, 0]
    )  # per signer monitoring, first one is how many are ok, second one is how many detections failed
    start = time.time()

    with make_landmarker() as landmarker:
        for class_dir in class_dirs:
            files = list_images(class_dir)
            if limit:
                files = files[:: max(1, len(files) // limit)]

            success = failed = 0
            for file in files:
                sid = signer_id(file)
                img = cv2.imread(str(file))

                if img is None:
                    failed += 1
                    start[sid][1] += 1
                    continue

                h, w = img.shape[:2]

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lm_array = landmarks_to_array(result.hand_landmarks[0])
                    lm_array[:, 1] *= h / w
                    X_list.append(lm_array)
                    labels.append(class_dir.name)
                    img_paths.append(str(file))
                    signers.append(sid)
                    success += 1
                    stats[sid][0] += 1
                else:
                    failed += 1
                    start[sid][1] += 1

            total = success + failed
            print(
                f"{class_dir.name:>3}: {success}/{total} detected "
                f"({100 * failed / max(total, 1):.1f}% failed)"
                f"[{time.time() - start:.0f}s]"
            )

    print("\nPer-signer detection:")
    for sid in sorted(stats):
        success, failed = stats[sid]
        print(
            f"  P{sid:>2}: {success}/{success + failed} detected "
            f"({100 * failed / max(success):.1f}% failed)"
        )

    X = np.array(X_list, dtype=np.float32)
    np.savez_compressed(
        paths.RAW_LANDMARKS_HG,
        X=X,
        y=np.array(labels),
        paths=np.array(img_paths),
        source=np.array([SOURCE] * len(X)),
        signer=np.array(signers),
    )
    print(f"\nSaved {paths.RAW_LANDMARKS_HG.name}: X shape = {X.shape}")


if __name__ == "__main__":
    main()
