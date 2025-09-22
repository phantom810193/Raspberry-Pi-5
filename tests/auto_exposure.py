"""Probe different exposure values across available V4L2 devices."""
from __future__ import annotations

import time
from glob import glob

import cv2

FOURCC = "YUYV"
WIDTH = 1280
HEIGHT = 720
EXPOSURE_STEPS = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
GAIN = 5.0  # optional tweak; drivers may ignore it
MIN_MEAN = 10.0


def _open_capture() -> tuple[cv2.VideoCapture, str]:
    """Return an opened ``VideoCapture`` plus the device path."""
    candidates = sorted(glob("/dev/video*"))
    last_error = ""
    for device in candidates:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_GAIN, float(GAIN))
        time.sleep(0.2)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, device
        last_error = f"{device} returned empty frame"
        cap.release()
    raise RuntimeError(last_error or "Could not open any /dev/video* candidate")


def main() -> None:
    cap, device = _open_capture()
    print(f"Using device: {device}")

    try:
        best_frame = None
        best_exposure = None
        for exposure in EXPOSURE_STEPS:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            time.sleep(0.3)
            ok, frame = cap.read()
            mean_value = float(frame.mean()) if ok and frame is not None else None
            print(f"exposure={exposure}, ok={ok}, mean={mean_value}")
            if ok and frame is not None and mean_value is not None and mean_value > MIN_MEAN:
                best_frame = frame
                best_exposure = exposure
                break

        if best_frame is not None:
            path = f"debug_exposure_{best_exposure}.jpg"
            cv2.imwrite(path, best_frame)
            print(f"Saved {path}")
        else:
            print("No exposure produced sufficient brightness; consider increasing lighting or extending EXPOSURE_STEPS.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
