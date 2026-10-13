"""Run an isolated-sign YOLO detector from a webcam and speak stable labels.

Example:
    python realtime_sign_to_speech.py --weights runs/nsl_yolo26/yolo26s_baseline/weights/best.pt

Press q to stop. This is a prototype: it recognises only labels included in training.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import pyttsx3
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained best.pt weights.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.65, help="Minimum detection confidence.")
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds before repeating a label.")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def best_label(result) -> tuple[str | None, float]:
    """Return only the most confident detected sign in this frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    top_index = int(result.boxes.conf.argmax().item())
    class_id = int(result.boxes.cls[top_index].item())
    confidence = float(result.boxes.conf[top_index].item())
    return result.names[class_id], confidence


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1 or allow camera permission.")

    speaker = pyttsx3.init()
    speaker.setProperty("rate", 165)
    recent_labels: deque[str] = deque(maxlen=args.stable_frames)
    last_spoken_label: str | None = None
    last_spoken_at = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            # YOLO returns boxes, class IDs, and confidence values for this single frame.
            result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            annotated = result.plot()
            label, confidence = best_label(result)

            if label is not None:
                recent_labels.append(label)
            else:
                recent_labels.clear()

            # Speak only after identical labels persist across consecutive frames.
            stable = (
                len(recent_labels) == args.stable_frames
                and len(set(recent_labels)) == 1
            )
            now = time.monotonic()
            if stable and (label != last_spoken_label or now - last_spoken_at >= args.cooldown):
                speaker.say(label.replace("_", " "))
                speaker.runAndWait()
                last_spoken_label, last_spoken_at = label, now

            status = "Waiting for a stable sign" if not stable else f"Speaking: {label} ({confidence:.0%})"
            cv2.putText(annotated, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("NSL YOLO26 sign-to-speech — press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
