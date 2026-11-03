import time
import cv2

CAMERA_INDEX = 0
MIRROR = True
WINDOW = "App (q to quit)"


def process(frame):
    """Receives a BGR frame, returns the frame to display."""
    return frame


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    prev = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, stopping.")
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            frame = process(frame)

            # fps smoothing
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / (now - prev)) # weighted avg
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()