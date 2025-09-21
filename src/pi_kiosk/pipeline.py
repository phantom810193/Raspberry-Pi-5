"""End-to-end pipeline orchestrating detection, persistence and advert generation."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
from .detection import DetectionConfig, FaceIdentifier, SimulatedFaceIdentifier


class Identifier(Protocol):
    """Protocol implemented by both the real and simulated identifiers."""

    def identify(self, image: np.ndarray) -> Iterable[str]:
        ...


@dataclass
class PipelineConfig:
    db_path: Path
    model_dir: Optional[Path] = None
    cooldown_seconds: int = 5
    simulated_member_ids: Optional[Tuple[str, ...]] = None
    classifier_path: Optional[Path] = None


class AdvertisementPipeline:
    """High level abstraction wrapping all kiosk behaviour."""

    def __init__(self, config: PipelineConfig, identifier: Optional[Identifier] = None):
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

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process a frame from the camera returning a new advert message if any."""
        member_ids = list(self.identifier.identify(frame))
        message: Optional[str] = None
        for member_id in member_ids:
            current_time = time.time()
            last_seen = self._id_last_seen.get(member_id, 0)
            if current_time - last_seen < self.config.cooldown_seconds:
                continue
            self._id_last_seen[member_id] = current_time
            message = self._handle_member(member_id)
        return message

    def simulate_member(self, member_id: str) -> str:
        """Public helper used by tests and demos to inject a member event."""
        with self._lock:
            return self._handle_member(member_id)

    def latest_message(self) -> Tuple[str, Optional[str], Optional[datetime]]:
        with self._lock:
            return self._latest_message, self._latest_id, self._latest_timestamp

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _handle_member(self, member_id: str) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        database.register_member(self.conn, member_id, timestamp)
        transactions = [
            advertising.Transaction(item=row["item"], amount=row["amount"], timestamp=row["timestamp"])
            for row in database.get_transactions(self.conn, member_id)
        ]
        message = advertising.generate_message(member_id, transactions)
        with self._lock:
            self._latest_message = message
            self._latest_id = member_id
            self._latest_timestamp = datetime.now(timezone.utc)
        return message


def create_pipeline(config: PipelineConfig) -> AdvertisementPipeline:
    """Factory that respects ``simulated_member_ids``."""
    identifier: Identifier
    if config.simulated_member_ids is not None:
        identifier = SimulatedFaceIdentifier(config.simulated_member_ids)
    else:
        identifier = None  # type: ignore
    return AdvertisementPipeline(config, identifier=identifier)


def camera_loop(pipeline: AdvertisementPipeline, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
    """Continuously read from the camera and feed frames into the pipeline."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV (cv2) is not installed; camera loop is unavailable")

    # ``libcamerify`` exposes the CSI camera as a V4L2 device. Request a format that
    # OpenCV can decode directly; ``YUYV`` works across libcamera builds and keeps the
    # conversion cost small compared to raw Bayer data.
    capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                LOGGER.warning("Failed to read frame from camera index %s", camera_index)
                time.sleep(1)
                continue
            if frame is None:
                LOGGER.warning("Camera index %s returned empty frame", camera_index)
                time.sleep(0.1)
                continue

            # Frames can come back as 2-channel YUYV or already expanded to 3-channel.
            # Normalise them to BGR for the downstream pipeline.
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (1, 2)):
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            elif frame.ndim == 3 and frame.shape[2] != 3:
                LOGGER.warning(
                    "Unexpected frame shape from camera index %s: %s", camera_index, frame.shape
                )
                time.sleep(0.1)
                continue
            pipeline.process_frame(frame)
            time.sleep(0.1)
    finally:
        capture.release()


LOGGER = logging.getLogger(__name__)
