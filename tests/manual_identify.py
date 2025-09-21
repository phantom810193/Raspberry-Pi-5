"""Manual camera capture test that prints face detection results."""
from __future__ import annotations

import sys
import time
from glob import glob
from pathlib import Path

import cv2
import face_recognition

from pi_kiosk.detection import DetectionConfig, FaceIdentifier

# Camera tuning -------------------------------------------------------------
WIDTH = 1280
HEIGHT = 720
FOURCC = "YUYV"
CANDIDATE_DEVICES = sorted(glob("/dev/video*"))  # libcamerify creates a temp device
EXPOSURE_US = 20_000
GAIN = 5.0


def _open_capture() -> tuple[cv2.VideoCapture, str]:
    """Return an opened ``VideoCapture`` and the path used."""
    last_error = ""
    for device in CANDIDATE_DEVICES:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        # Try to set exposure/gain if driver honours it (values are device dependent).
        cap.set(cv2.CAP_PROP_EXPOSURE, float(EXPOSURE_US))
        cap.set(cv2.CAP_PROP_GAIN, float(GAIN))
        time.sleep(0.3)  # give sensor time to settle
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, device
        last_error = f"{device} returned empty frame"
        cap.release()
    raise RuntimeError(last_error or "Could not open any /dev/video* candidate")


def main() -> int:
    config = DetectionConfig(
        shape_predictor_path=Path("models/shape_predictor_68_face_landmarks.dat"),
        face_recognition_model_path=Path("models/dlib_face_recognition_resnet_model_v1.dat"),
        classifier_path=Path("models/face_classifier.pkl"),
    )
    identifier = FaceIdentifier(config)

    cap, device_path = _open_capture()
    print(f"Using device: {device_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("capture ok? False")
        return 1

    print("capture ok? True")
    print("raw shape:", frame.shape, "dtype:", frame.dtype)

    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (1, 2)):
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        print("converted to BGR, shape:", frame.shape)
    elif frame.ndim == 3 and frame.shape[2] != 3:
        print(f"Unexpected frame shape: {frame.shape}")
        return 1

    # Save for inspection
    cv2.imwrite("debug_capture.jpg", frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    descriptors = identifier.encode_faces(rgb)
    print("encoded faces:", len(descriptors))
    ids = identifier.identify(rgb)
    print("identify result:", ids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
