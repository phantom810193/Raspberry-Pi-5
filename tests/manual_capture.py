import cv2

def capture_frame(index: int = 0, width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")

    fourccs = ["YUYV", "RGB3", "MJPG", None]
    for code in fourccs:
        if code:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        print(f"attempt fourcc={code!r} -> ok={ok}")
        if ok:
            print("shape:", getattr(frame, "shape", None))
            print("dtype:", getattr(frame, "dtype", None))
            cap.release()
            return frame

    cap.release()
    raise RuntimeError("Camera did not return a frame with supported format")

if __name__ == "__main__":
    capture_frame()
