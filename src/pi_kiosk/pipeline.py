"""End-to-end pipeline orchestrating detection, persistence and advert generation."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Tuple

try:
    import cv2  # type: ignore

    OPENCV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    OPENCV_AVAILABLE = False

import numpy as np

from . import advertising, database
from .ai_client import AIClient, build_context_from_transactions
from .detection import DetectionConfig, FaceIdentifier, SimulatedFaceIdentifier


FOURCC = "YUYV"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
EXPOSURE_CHOICES = (800.0, 600.0, 400.0)
MIN_MEAN = 15.0
GAIN = 5.0
CANDIDATE_GLOB = "/dev/video*"


class Identifier(Protocol):
    """Protocol implemented by both the real and simulated identifiers."""

    def identify(self, image: np.ndarray) -> Iterable[str]:
        ...


@dataclass
class PipelineConfig:
    db_path: Path
    model_dir: Optional[Path] = None
    cooldown_seconds: int = 2
    idle_reset_seconds: Optional[int] = 2
    simulated_member_ids: Optional[Tuple[str, ...]] = None
    classifier_path: Optional[Path] = None


class AdvertisementPipeline:
    """High level abstraction wrapping all kiosk behaviour."""

    def __init__(
        self,
        config: PipelineConfig,
        identifier: Optional[Identifier] = None,
        ai_client: Optional[AIClient] = None,
    ):
        self.config = config
        self.conn = database.connect(config.db_path)
        database.initialize_db(self.conn)
        database.ensure_sample_transactions(self.conn)

        if identifier is None:
            if config.model_dir is None:
                raise ValueError("model_dir must be specified when using the real FaceIdentifier")
            model_dir = Path(config.model_dir)
            classifier_path = config.classifier_path
            if classifier_path is None:
                default_classifier = model_dir / "face_classifier.pkl"
                if default_classifier.exists():
                    classifier_path = default_classifier
            detection_config = DetectionConfig(
                shape_predictor_path=model_dir / "shape_predictor_68_face_landmarks.dat",
                face_recognition_model_path=model_dir / "dlib_face_recognition_resnet_model_v1.dat",
                classifier_path=classifier_path,
            )
            identifier = FaceIdentifier(detection_config)
        self.identifier = identifier

        self._latest_message = "等待辨識中..."
        self._latest_id: Optional[str] = None
        self._latest_timestamp: Optional[datetime] = None
        self._id_last_seen: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._last_detection_time: Optional[float] = None
        self._ai_client = ai_client or AIClient()
        self._ai_busy = False

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process a frame from the camera returning a new advert message if any."""
        current_time = time.time()
        member_ids = list(self.identifier.identify(frame))
        if member_ids:
            with self._lock:
                self._last_detection_time = current_time
        message: Optional[str] = None
        for member_id in member_ids:
            last_seen = self._id_last_seen.get(member_id, 0)
            if current_time - last_seen < self.config.cooldown_seconds:
                continue
            self._id_last_seen[member_id] = current_time
            message = self._handle_member(member_id, current_time=current_time)
        self._maybe_reset_idle(time.time())
        return message

    def simulate_member(self, member_id: str) -> str:
        """Public helper used by tests and demos to inject a member event."""
        with self._lock:
            return self._handle_member(member_id)

    def latest_message(self) -> Tuple[str, Optional[str], Optional[datetime]]:
        with self._lock:
            return self._latest_message, self._latest_id, self._latest_timestamp

    def ai_busy(self) -> bool:
        with self._lock:
            return self._ai_busy

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _handle_member(self, member_id: str, *, current_time: Optional[float] = None) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        database.register_member(self.conn, member_id, timestamp)
        transactions = [
            advertising.Transaction(item=row["item"], amount=row["amount"], timestamp=row["timestamp"])
            for row in database.get_transactions(self.conn, member_id)
        ]

        with self._lock:
            self._latest_message = "辨識完成，讀取購買紀錄中…"
            self._latest_id = member_id
            self._latest_timestamp = datetime.now(timezone.utc)
            self._last_detection_time = current_time if current_time is not None else time.time()

        message = advertising.generate_message(member_id, transactions)
        ai_message: Optional[str] = None
        if transactions:
            context_transactions = [
                {"item": tx.item, "amount": tx.amount, "timestamp": tx.timestamp}
                for tx in transactions
            ]
            context = build_context_from_transactions(context_transactions)
            if context_transactions:
                context.setdefault("商品", context_transactions[0]["item"])
                context.setdefault("價格", str(context_transactions[0]["amount"]))
            with self._lock:
                self._latest_message = "產生廣告資訊中…"
                self._ai_busy = True
            try:
                ai_message = self._ai_client.generate(member_id, context)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("AI generation raised %s for member %s", exc, member_id)
                ai_message = None
            finally:
                with self._lock:
                    self._ai_busy = False

        final_message = ai_message or message

        with self._lock:
            self._latest_message = final_message
            self._latest_id = member_id
            self._latest_timestamp = datetime.now(timezone.utc)
            self._last_detection_time = current_time if current_time is not None else time.time()
        return final_message

    def add_transactions(self, member_id: str, records: Iterable[Tuple[str, float, str]]) -> int:
        """Append transactions for an existing member."""
        with self._lock:
            return database.insert_transactions(self.conn, member_id, records)

    def _maybe_reset_idle(self, current_time: float) -> None:
        idle_seconds = self.config.idle_reset_seconds
        if idle_seconds is None or idle_seconds <= 0:
            return
        with self._lock:
            if self._latest_id is None:
                return
            if self._last_detection_time is None:
                return
            if current_time - self._last_detection_time < idle_seconds:
                return
            self._latest_message = "等待辨識中..."
            self._latest_id = None
            self._latest_timestamp = None
            self._last_detection_time = None


def create_pipeline(config: PipelineConfig, ai_client: Optional[AIClient] = None) -> AdvertisementPipeline:
    """Factory that respects ``simulated_member_ids``."""
    identifier: Identifier
    if config.simulated_member_ids is not None:
        identifier = SimulatedFaceIdentifier(config.simulated_member_ids)
    else:
        identifier = None  # type: ignore
    return AdvertisementPipeline(config, identifier=identifier, ai_client=ai_client)


def _candidate_devices(primary_index: int) -> Tuple[str, ...]:
    devices = sorted(glob(CANDIDATE_GLOB))
    primary = f"/dev/video{primary_index}"
    if primary in devices:
        devices.remove(primary)
        devices.insert(0, primary)
    return tuple(devices)


def _normalise_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (1, 2)):
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    if frame.ndim == 3 and frame.shape[2] != 3:
        return None
    return frame


def _open_capture(
    camera_index: int,
    width: Optional[int],
    height: Optional[int],
) -> Tuple["cv2.VideoCapture", str, np.ndarray]:
    last_error = ""
    candidates = _candidate_devices(camera_index)
    if not candidates:
        raise RuntimeError("Could not open any /dev/video* candidate")

    target_width = float(width or DEFAULT_WIDTH)
    target_height = float(height or DEFAULT_HEIGHT)

    for device in candidates:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        for exposure in EXPOSURE_CHOICES:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            time.sleep(0.2)
            for _ in range(3):
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                mean_value = float(frame.mean())
                LOGGER.info("device=%s, exposure=%s, mean=%.2f", device, exposure, mean_value)
                if mean_value > MIN_MEAN:
                    return cap, device, frame

        cap.set(cv2.CAP_PROP_EXPOSURE, float(EXPOSURE_CHOICES[0]))
        cap.set(cv2.CAP_PROP_GAIN, float(GAIN))
        time.sleep(0.3)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, device, frame

        last_error = f"{device} returned empty frame despite exposure attempts"
        cap.release()

    raise RuntimeError(last_error or "Could not open any /dev/video* candidate")


def camera_loop(pipeline: AdvertisementPipeline, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
    """Continuously read from the camera and feed frames into the pipeline."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV (cv2) is not installed; camera loop is unavailable")

    capture, device_path, frame = _open_capture(camera_index, width, height)
    LOGGER.info("Using video device: %s", device_path)

    initial_bgr = _normalise_frame(frame)
    if initial_bgr is None:
        LOGGER.warning("Initial frame from %s had unexpected shape", device_path)
    else:
        rgb_frame = cv2.cvtColor(initial_bgr, cv2.COLOR_BGR2RGB)
        pipeline.process_frame(rgb_frame)

    try:
        while True:
            if pipeline.ai_busy():
                time.sleep(1.0)
                continue
            ok, frame = capture.read()
            if not ok:
                LOGGER.warning("Failed to read frame from camera index %s", camera_index)
                time.sleep(1)
                continue
            if frame is None:
                LOGGER.warning("Camera index %s returned empty frame", camera_index)
                time.sleep(0.1)
                continue

            normalised = _normalise_frame(frame)
            if normalised is None:
                LOGGER.warning("Unexpected frame shape from camera index %s", camera_index)
                time.sleep(0.1)
                continue

            rgb_frame = cv2.cvtColor(normalised, cv2.COLOR_BGR2RGB)
            pipeline.process_frame(rgb_frame)
            time.sleep(0.1)
    finally:
        capture.release()


LOGGER = logging.getLogger(__name__)
