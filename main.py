from asl.inference import load_model, predict
from asl.hand_landmarker import make_landmarker, landmarks_to_array, draw_hand
from asl import paths
from asl.reference import first_clean_image_per_letter, build_reference_grid
from asl.translation import FingerSpeller
import cv2
import time
import mediapipe as mp
import numpy as np
from collections import deque

CAMERA_INDEX = 0
WINDOW = "ASL Detection (q to quit)"
SMOOTH_FRAMES = 8
DATASET_PATHS = [paths.CLEAN_FEATURES, paths.CLEAN_FEATURES_HG]


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    start = prev = time.perf_counter()
    fps = 0.0

    model, classes, mean, std = load_model(paths.MODEL_COMBINED)
    class_text = ""

    # so that the displayed letter doesn't flicker too much
    buffer = deque(maxlen=SMOOTH_FRAMES)

    # reference grid
    data_path_idx = 0
    grid, cell_post = build_reference_grid(
        first_clean_image_per_letter(DATASET_PATHS[data_path_idx])
    )
    cv2.namedWindow("Reference")
    cv2.moveWindow("Reference", 700, 0)

    fingerspeller = FingerSpeller()
    letter = None
    confidence = 0.0
    text = ""

    try:
        with make_landmarker(video=True) as landmarker:
            while True:
                res, frame = cap.read()
                if not res:
                    print("Frame read failed, stopping.")
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

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
                    probs, ok = predict(
                        model, mean, std, landmark_mat, frame_h=h, frame_w=w
                    )

                    # display result
                    if ok:
                        buffer.append(probs)
                        avg = np.mean(buffer, axis=0)
                        letter = classes[avg.argmax()]
                        confidence = float(avg.max())
                        class_text = f"{letter} ({confidence * 100:.0f}%)"
                    else:
                        buffer.clear()
                        letter = None
                        confidence = 0.0
                        class_text = "Move Closer"
                else:
                    buffer.clear()
                    letter = None
                    confidence = 0.0
                    class_text = "No Hand"

                # Smoothed FPS
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1 / (now - prev))
                prev = now

                output = fingerspeller.update(letter, confidence)
                if output:
                    text += output

                # drawing
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    class_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    text[-30:],
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                # progress bar to commit letter
                bar_w = int(200 * fingerspeller.progress)
                cv2.rectangle(frame, (10, 75), (210, 90), (80, 80, 80), 1)    
                cv2.rectangle(frame, (10, 75), (10 + bar_w, 90), (0, 255, 0), -1)

                cv2.imshow(WINDOW, frame)
                cv2.imshow("Reference", grid)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    data_path_idx += 1
                    if data_path_idx > len(DATASET_PATHS) - 1:
                        data_path_idx = 0
                    grid, cell_post = build_reference_grid(
                        first_clean_image_per_letter(DATASET_PATHS[data_path_idx])
                    )

                # typing controls
                if key == 32:                   
                    text += " "
                elif key == 8:                  
                    text = text[:-1]
                elif key == ord("c"):
                    text = ""

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
