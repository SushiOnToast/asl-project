from asl.inference import load_model, predict
from asl.hand_landmarker import make_landmarker, landmarks_to_array, draw_hand
from asl import paths
import cv2
import time
import mediapipe as mp
import numpy as np
from collections import deque

CAMERA_INDEX = 0
WINDOW = "ASL Detection (q to quit)"
SMOOTH_FRAMES = 8

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    start = prev = time.perf_counter()
    fps = 0.0

    model, classes, mean, std = load_model(paths.MODEL_MIRROR)
    class_text = ""

    # so that the displayed letter doesn't flicker too much
    buffer = deque(maxlen=SMOOTH_FRAMES)

    flip_input = True

    try:
        with make_landmarker(video=True) as landmarker:
            while True:
                res, frame = cap.read()
                if not res:
                    print("Frame read failed, stopping.")
                    break
                if flip_input:
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
                    h, w = frame.shape[:2]
                    probs, ok = predict(model, mean, std, landmark_mat, frame_h=h, frame_w=w)

                    # display result
                    if ok:
                        buffer.append(probs)
                        avg = np.mean(buffer, axis=0) 
                        class_text = f"{classes[avg.argmax()]} ({avg.max() * 100:.0f}%)"
                    else:
                        buffer.clear()
                        class_text = "Move Closer"
                else:
                    buffer.clear()
                    class_text = "No Hand"


                # Smoothed FPS
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1 / (now - prev))
                prev = now

                # drawing
                cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Flip input: {flip_input}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    flip_input = not flip_input
                    buffer.clear()
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()