#!/usr/bin/env python3
"""Extract face embeddings for each identity using face_recognition."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import face_recognition
import numpy as np

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/faces"), help="Root directory containing face images")
    parser.add_argument("--output", type=Path, default=Path("models/known_faces.npz"), help="Output path for embeddings")
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("models/labels.json"),
        help="Optional JSON file storing label list for reference",
    )
    parser.add_argument("--upsample", type=int, default=1, help="Number of times to upsample when finding faces")
    parser.add_argument(
        "--jitters",
        type=int,
        default=1,
        help="Number of times to re-sample the face when computing encodings",
    )
    parser.add_argument("--tolerance", type=int, default=5, help="Minimum images per person (warning only)")
    return parser.parse_args(argv)


@dataclass
class PersonEmbedding:
    label: str
    embeddings: List[np.ndarray]


def discover_people(root: Path) -> Dict[str, PersonEmbedding]:
    people: Dict[str, PersonEmbedding] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        embeddings: List[np.ndarray] = []
        people[person_dir.name] = PersonEmbedding(label=person_dir.name, embeddings=embeddings)
    return people


def gather_images(person_dir: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(person_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return images


def encode_image(image_path: Path, upsample: int, jitters: int) -> Tuple[List[np.ndarray], str | None]:
    try:
        image = face_recognition.load_image_file(str(image_path))
    except Exception as exc:  # pragma: no cover - depends on local files
        return [], f"無法載入 {image_path}: {exc}"

    encodings = face_recognition.face_encodings(image, num_jitters=jitters, num_upsamples=upsample)
    if not encodings:
        return [], f"{image_path} 未找到臉部特徵"
    if len(encodings) > 1:
        LOGGER.warning("%s 偵測到多於一張臉，只取第一個", image_path)
    return [np.array(encodings[0])], None


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_dir: Path = args.input
    output_path: Path = args.output
    labels_path: Path = args.labels_output

    if not input_dir.exists():
        LOGGER.error("輸入目錄 %s 不存在", input_dir)
        return 1

    people = discover_people(input_dir)
    if not people:
        LOGGER.error("在 %s 下找不到任何身份資料夾", input_dir)
        return 1

    failures: List[str] = []

    all_labels: List[str] = []
    all_embeddings: List[np.ndarray] = []

    for person_name, person in people.items():
        images = gather_images(input_dir / person_name)
        if not images:
            LOGGER.warning("%s 沒有影像檔，略過", person_name)
            continue
        LOGGER.info("處理 %s，共 %d 張照片", person_name, len(images))
        person_vectors: List[np.ndarray] = []
        for image_path in images:
            encoding, error = encode_image(image_path, args.upsample, args.jitters)
            if error:
                failures.append(error)
                continue
            person_vectors.extend(encoding)
        if not person_vectors:
            LOGGER.warning("%s 沒有可用的臉部向量", person_name)
            continue
        if len(person_vectors) < args.tolerance:
            LOGGER.warning("%s 只有 %d 筆向量，建議補拍更多照片", person_name, len(person_vectors))
        mean_vector = np.mean(person_vectors, axis=0)
        all_labels.append(person_name)
        all_embeddings.append(mean_vector.astype(np.float32))

    if not all_labels:
        LOGGER.error("沒有任何成功的臉部向量，無法輸出模型")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, embeddings=np.stack(all_embeddings), labels=np.array(all_labels))
    labels_path.write_text(json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("已輸出 %d 位身份的平均特徵到 %s", len(all_labels), output_path)
    if failures:
        LOGGER.info("共有 %d 張影像失敗，詳見下方：", len(failures))
        for failure in failures:
            LOGGER.info("- %s", failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
