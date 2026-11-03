import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).parent / "hand_landmarker.task"
CAMERA_INDEX = 0
MIRROR = True
WINDOW = "Landmarker (q quit, p print)"

CHAINS = [
    [0, 1, 2, 3, 4],        # thumb
    [5, 6, 7, 8],           # index
    [9, 10, 11, 12],        # middle
    [13, 14, 15, 16],       # ring
    [17, 18, 19, 20],       # pinky
    [0, 5, 9, 13, 17, 0],   # palm, closed by ending back at 0
]

CONNECTIONS = [pair for chain in CHAINS for pair in zip(chain, chain[1:])]


def make_landmarker():
    if not MODEL.exists():
        raise SystemExit(f"Model file not found: {MODEL}")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(hand):
    """21 MediaPipe landmarks -> float32 array of shape (21, 3)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    start = prev = time.perf_counter()
    fps = 0.0
    shape_printed = False

    try:
        with make_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Frame read failed, stopping.")
                    break
                if MIRROR:
                    frame = cv2.flip(frame, 1)

                # Detect on the clean frame, before drawing anything on it
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.perf_counter() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                # landmark processing and drawing
                landmark_mat = None
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    landmark_mat = landmarks_to_array(hand)
                    draw_hand(frame, hand)
                    if not shape_printed:
                        print("Landmark array shape:", landmark_mat.shape)
                        shape_printed = True

                # Smoothed FPS
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1 / (now - prev))
                prev = now
                cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    if landmark_mat is None:
                        print("No hand detected.")
                    else:
                        print(f"Wrist: {landmark_mat[0]}  Index tip: {landmark_mat[8]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()