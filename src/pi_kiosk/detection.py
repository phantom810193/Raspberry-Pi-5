"""Face detection utilities built on top of dlib.

The module exposes a :class:`FaceIdentifier` which converts the facial feature
vector produced by dlib into an anonymous but stable identifier by hashing the
128D descriptor.

When dlib or the required pre-trained models are unavailable, the
:class:`SimulatedFaceIdentifier` can be used to drive demos and automated tests
without camera hardware.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency on Raspberry Pi
    import dlib

    DLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - executed on developer machines without dlib
    dlib = None  # type: ignore
    DLIB_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration required to construct a :class:`FaceIdentifier`."""

    shape_predictor_path: Path
    face_recognition_model_path: Path
    cnn_detector_path: Optional[Path] = None
    classifier_path: Optional[Path] = None


@dataclass
class ClassifierModel:
    """Lightweight nearest-neighbour classifier built from averaged embeddings."""

    labels: List[str]
    embeddings: np.ndarray
    distance_threshold: float

    def match(self, descriptor: np.ndarray) -> Optional[str]:
        if not len(self.labels):
            return None
        descriptor = descriptor.astype(np.float32, copy=False)
        distances = np.linalg.norm(self.embeddings - descriptor, axis=1)
        best_index = int(np.argmin(distances))
        if distances[best_index] <= self.distance_threshold:
            return self.labels[best_index]
        return None


class ModelMissingError(RuntimeError):
    """Raised when a required model file could not be found."""


class FaceIdentifier:
    """Wrap dlib's models to provide hashed member IDs for detected faces."""

    def __init__(self, config: DetectionConfig):
        if not DLIB_AVAILABLE:
            raise ImportError("dlib is not installed; use SimulatedFaceIdentifier instead")

        self.config = config
        self._detector = self._load_detector(config.cnn_detector_path)
        self._shape_predictor = self._load_shape_predictor(config.shape_predictor_path)
        self._face_recognition = self._load_face_recognition_model(config.face_recognition_model_path)
        self._classifier = self._load_classifier(config.classifier_path)

    @staticmethod
    def _load_detector(cnn_detector_path: Optional[Path]):
        if cnn_detector_path:
            path = Path(cnn_detector_path)
            if not path.exists():
                raise ModelMissingError(
                    "CNN detector model not found at %s. Download 'mmod_human_face_detector.dat'" % path
                )
            return dlib.cnn_face_detection_model_v1(str(path))
        return dlib.get_frontal_face_detector()

    @staticmethod
    def _load_shape_predictor(path: Path):
        path = Path(path)
        if not path.exists():
            raise ModelMissingError(
                "Shape predictor model not found at %s. Download 'shape_predictor_68_face_landmarks.dat'"
                " from the dlib model zoo." % path
            )
        return dlib.shape_predictor(str(path))

    @staticmethod
    def _load_face_recognition_model(path: Path):
        path = Path(path)
        if not path.exists():
            raise ModelMissingError(
                "Face recognition model not found at %s. Download 'dlib_face_recognition_resnet_model_v1.dat'"
                " from the dlib model zoo." % path
            )
        return dlib.face_recognition_model_v1(str(path))

    @staticmethod
    def _load_classifier(path: Optional[Path]) -> Optional[ClassifierModel]:
        if path is None:
            return None
        resolved = Path(path)
        if not resolved.exists():
            raise ModelMissingError(f"Classifier file not found at {resolved}. Run the training scripts first.")
        with resolved.open("rb") as fh:
            payload = pickle.load(fh)

        embeddings = np.array(payload.get("embeddings"), dtype=np.float32)
        labels = list(payload.get("labels", []))
        threshold = float(payload.get("distance_threshold", 0.45))

        if embeddings.size == 0 or not labels:
            raise ModelMissingError(f"Classifier at {resolved} does not contain embeddings/labels")
        if embeddings.shape[0] != len(labels):
            raise ModelMissingError(
                f"Classifier at {resolved} has mismatched embeddings ({embeddings.shape[0]}) and labels ({len(labels)})"
            )

        LOGGER.info("Loaded classifier with %d identities from %s", len(labels), resolved)
        return ClassifierModel(labels=labels, embeddings=embeddings, distance_threshold=threshold)

    def encode_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Return 128D descriptors for the detected faces in ``image``."""
        detections = self._detector(image, 1)
        descriptors: List[np.ndarray] = []

        for detection in detections:
            # ``cnn_face_detection_model_v1`` returns mmod rectangles
            rect = detection.rect if hasattr(detection, "rect") else detection
            shape = self._shape_predictor(image, rect)
            descriptor = self._face_recognition.compute_face_descriptor(image, shape)
            descriptors.append(np.array(descriptor))
        return descriptors

    def identify(self, image: np.ndarray) -> List[str]:
        """Return anonymous IDs for the faces detected in ``image``."""
        ids: List[str] = []
        for descriptor in self.encode_faces(image):
            label: Optional[str] = None
            if self._classifier is not None:
                label = self._classifier.match(descriptor)
                if label is None:
                    LOGGER.debug("Descriptor unmatched; fall back to hashed ID")
            if label is None:
                label = hash_descriptor(descriptor)
            ids.append(label)
        if ids:
            LOGGER.debug("Identified faces: %s", ids)
        return ids


class SimulatedFaceIdentifier:
    """Deterministically generate identifiers without inspecting the image."""

    def __init__(self, simulated_ids: Optional[Iterable[str]] = None):
        self._ids = list(simulated_ids or [])
        self._index = 0

    def identify(self, image: np.ndarray) -> List[str]:  # pragma: no cover - trivial
        if not self._ids:
            return ["member-simulated"]
        value = self._ids[self._index % len(self._ids)]
        self._index += 1
        return [value]


def hash_descriptor(descriptor: np.ndarray) -> str:
    """Convert a 128D face descriptor into a short stable identifier."""
    byte_view = descriptor.astype(np.float32).tobytes()
    digest = hashlib.sha1(byte_view).hexdigest()
    return "member-" + digest[:12]
