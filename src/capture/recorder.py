"""
Simple capture tool for recording raw frames for future training.

Usage (from project root):
    python src/capture/recorder.py

Controls:
    q - quit
    s - save current frame as JPG under data/raw/<label>/
"""

import time
from pathlib import Path

import cv2


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_raw = project_root / "data" / "raw"

    label = "custom"  # TODO: change this before recording (e.g. "ok", "rock")
    out_dir = data_raw / label
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("📷 RT-Gesture3D Capture Recorder")
    print(f"Output directory: {out_dir}")
    print("Press 's' to save frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        cv2.putText(
            frame,
            f"Label: {label} | 's' = save, 'q' = quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("RT-Gesture3D - Capture Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("👋 Exiting.")
            break
        elif key == ord("s"):
            ts = int(time.time() * 1000)
            out_path = out_dir / f"{label}_{ts}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"💾 Saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
