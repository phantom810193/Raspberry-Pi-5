"""End-to-end pipeline orchestrating detection, persistence and advert generation."""
from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

try:
    import cv2  # type: ignore

    OPENCV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    OPENCV_AVAILABLE = False

import numpy as np

from . import advertising, database
from .ai_client import AIClient, build_context_from_transactions
from .detection import ClassifierModel, DetectionConfig, FaceIdentifier, FaceMatch, SimulatedFaceIdentifier


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
    use_trained_classifier: bool = True
    auto_enroll_first_face: bool = False
    auto_enroll_threshold: float = 0.45
    store_face_snapshot: bool = False


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
            classifier_path = config.classifier_path if config.use_trained_classifier else None
            if config.use_trained_classifier and classifier_path is None:
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

        self._supports_face_identifier = hasattr(self.identifier, "identify_faces") and hasattr(self.identifier, "set_classifier")
        self._base_classifier: Optional[ClassifierModel] = None
        self._registered_face_ids: set[str] = set()
        self._base_classifier_labels: set[str] = set()
        self._auto_enroll_done = False
        self._enrolled_sources: Dict[str, str] = {}

        if self._supports_face_identifier:
            get_classifier = getattr(self.identifier, "get_classifier", lambda: None)
            base_classifier = get_classifier()
            if base_classifier is not None and self.config.use_trained_classifier:
                self._base_classifier = self._clone_classifier(base_classifier)
            if not self.config.use_trained_classifier and hasattr(self.identifier, "set_classifier"):
                self.identifier.set_classifier(None)
            self._reload_classifier_from_db()

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
        member_sources: Dict[str, str] = {}

        if self._supports_face_identifier:
            matches = self.identifier.identify_faces(frame)
            if matches:
                matches.sort(
                    key=lambda m: (m.location.bottom - m.location.top) * (m.location.right - m.location.left),
                    reverse=True,
                )
                largest = matches[0]
                if len(matches) > 1:
                    LOGGER.debug(
                        "Filtered %d faces; processing only the largest face labelled %s",
                        len(matches) - 1,
                        largest.label,
                    )
                matches = [largest]
            auto_sources = self._maybe_auto_enroll(matches, frame)
            if auto_sources:
                member_sources.update(auto_sources)
            for match in matches:
                member_sources.setdefault(match.label, self._categorize_match(match))
            member_ids = [match.label for match in matches]
        else:
            matches = []
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
            source = member_sources.get(member_id)
            message = self._handle_member(member_id, current_time=current_time, source=source)
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
    def _handle_member(self, member_id: str, *, current_time: Optional[float] = None, source: Optional[str] = None) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        database.register_member(self.conn, member_id, timestamp, source=source, updated_at=timestamp)
        database.update_member_metadata(self.conn, member_id, source=source, updated_at=timestamp)
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

    def add_face_feature(
        self,
        member_id: str,
        descriptor: np.ndarray,
        *,
        snapshot: Optional[bytes] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """Persist a facial descriptor and refresh the active classifier."""

        if not self._supports_face_identifier:
            raise RuntimeError("Face feature management requires a FaceIdentifier")

        vector = np.asarray(descriptor, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("descriptor must be a 1D array")
        created = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

        database.register_member(self.conn, member_id, created, source="api", updated_at=created)
        database.store_face_feature(self.conn, member_id, vector.tobytes(), created, snapshot)
        database.update_member_metadata(self.conn, member_id, source="api", updated_at=created)
        self._reload_classifier_from_db()

    def remove_face_feature(self, member_id: str) -> bool:
        """Remove the descriptor for ``member_id``; returns True if deleted."""

        if not self._supports_face_identifier:
            return False
        removed = database.delete_face_feature(self.conn, member_id)
        if removed:
            self._reload_classifier_from_db()
        return removed

    def list_face_feature_ids(self) -> Tuple[str, ...]:
        """Return the member IDs that currently have stored descriptors."""

        if not self._supports_face_identifier:
            return tuple()
        return tuple(database.list_face_feature_ids(self.conn))

    @staticmethod
    def _clone_classifier(model: ClassifierModel) -> ClassifierModel:
        return ClassifierModel(
            labels=list(model.labels),
            embeddings=model.embeddings.astype(np.float32, copy=True),
            distance_threshold=float(model.distance_threshold),
        )

    def _reload_classifier_from_db(self) -> None:
        if not self._supports_face_identifier:
            return

        rows = database.get_face_features(self.conn)
        labels: List[str] = []
        descriptors: List[np.ndarray] = []
        enrolled_sources: Dict[str, str] = {}
        for row in rows:
            member_id = str(row["member_id"])
            descriptor_bytes = row["descriptor"]
            if descriptor_bytes is None:
                continue
            vector = np.frombuffer(descriptor_bytes, dtype=np.float32)
            if vector.size == 0:
                continue
            labels.append(member_id)
            descriptors.append(vector)
            source_value = row["source"] if "source" in row.keys() else None
            if source_value:
                enrolled_sources[member_id] = str(source_value)
            else:
                enrolled_sources[member_id] = "auto_enroll"

        self._registered_face_ids = set(labels)
        self._auto_enroll_done = bool(self._registered_face_ids)
        self._enrolled_sources = enrolled_sources

        db_classifier: Optional[ClassifierModel] = None
        if descriptors:
            stacked = np.stack(descriptors).astype(np.float32)
            db_classifier = ClassifierModel(
                labels=list(labels),
                embeddings=stacked,
                distance_threshold=float(self.config.auto_enroll_threshold),
            )

        active_classifier: Optional[ClassifierModel]
        if self.config.use_trained_classifier and self._base_classifier is not None:
            base_clone = self._clone_classifier(self._base_classifier)
            self._base_classifier_labels = set(base_clone.labels)
            if db_classifier is not None:
                active_classifier = ClassifierModel(
                    labels=list(base_clone.labels) + list(db_classifier.labels),
                    embeddings=np.vstack([base_clone.embeddings, db_classifier.embeddings]).astype(np.float32),
                    distance_threshold=float(base_clone.distance_threshold),
                )
            else:
                active_classifier = base_clone
        else:
            self._base_classifier_labels = set()
            active_classifier = db_classifier

        self.identifier.set_classifier(active_classifier)

    def _maybe_auto_enroll(self, matches: List[FaceMatch], frame: np.ndarray) -> Dict[str, str]:
        sources: Dict[str, str] = {}
        if not self.config.auto_enroll_first_face:
            return sources
        if not matches:
            return sources
        if self._auto_enroll_done:
            return sources

        first = matches[0]
        member_id = str(first.label)
        if member_id in self._registered_face_ids:
            self._auto_enroll_done = True
            sources[member_id] = self._enrolled_sources.get(member_id, "auto_enroll")
            return sources

        descriptor = first.descriptor
        if descriptor.ndim != 1:
            return sources

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        database.register_member(self.conn, member_id, created_at, source="auto_enroll", updated_at=created_at)

        snapshot_bytes: Optional[bytes] = None
        if self.config.store_face_snapshot:
            snapshot_bytes = self._encode_snapshot(frame, first)

        database.store_face_feature(self.conn, member_id, descriptor.astype(np.float32).tobytes(), created_at, snapshot_bytes)
        database.update_member_metadata(self.conn, member_id, source="auto_enroll", updated_at=created_at)
        self._reload_classifier_from_db()
        self._auto_enroll_done = True
        sources[member_id] = "auto_enroll"
        return sources

    def _categorize_match(self, match: FaceMatch) -> str:
        label = str(match.label)
        enrolled_source = self._enrolled_sources.get(label)
        if enrolled_source is not None:
            return enrolled_source
        if match.matched and label in self._base_classifier_labels:
            return "trained"
        if match.matched:
            return "trained"
        return "unknown"

    def _encode_snapshot(self, frame: np.ndarray, match: FaceMatch) -> Optional[bytes]:
        location = match.location
        height, width = frame.shape[:2]
        top = max(int(location.top), 0)
        bottom = min(int(location.bottom), height)
        left = max(int(location.left), 0)
        right = min(int(location.right), width)
        if top >= bottom or left >= right:
            return None
        face_region = frame[top:bottom, left:right]
        if face_region.size == 0:
            return None

        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - pillow should be available with face_recognition
            return None

        with io.BytesIO() as buffer:
            Image.fromarray(face_region).save(buffer, format="JPEG")
            return buffer.getvalue()


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
