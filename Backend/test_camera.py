"""
Hand Landmark Feature Extraction Pipeline
==========================================
Production-grade two-hand landmark extractor for Indian Sign Language (ISL)
classification. Extracts normalized, scale-invariant, rotation-robust 126-dim
feature vectors from raw images using MediaPipe Hand Landmarker.

Feature vector layout:
    [0:63]   → Hand 1 (left-most) : 21 landmarks × 3 (x, y, z) — wrist-normalized
    [63:126] → Hand 2 (right-most): 21 landmarks × 3 (x, y, z) — wrist-normalized
    [126]    → label
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore")

# ─────────────────────────── Configuration ────────────────────────────────── #

DATASET_PATH  = "dataset"
MODEL_PATH    = "hand_landmarker.task"
OUTPUT_CSV    = "isl_twohand_landmarks.csv"
NUM_HANDS     = 2
NUM_LANDMARKS = 21
FEATURE_DIM   = NUM_HANDS * NUM_LANDMARKS * 3   # 126
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE  = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ──────────────────────────── Logging Setup ────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Detector Setup ──────────────────────────────── #

def build_detector() -> vision.HandLandmarker:
    """Instantiate a MediaPipe HandLandmarker with production settings."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


# ────────────────────────── Feature Engineering ───────────────────────────── #

def _normalize_hand(hand_landmarks: list) -> Optional[np.ndarray]:
    """
    Normalize a single hand's 21 landmarks into a scale-invariant,
    translation-invariant 63-dim vector.

    Strategy
    --------
    1. **Translation**: subtract wrist (landmark 0) so wrist = origin.
    2. **Scale**      : divide by the Euclidean distance wrist → middle-finger MCP
                        (landmark 9) — more stable than wrist→middle-tip because
                        MCP position barely changes with finger curl.
    3. Result: each landmark becomes (delta_x/s, delta_y/s, delta_z/s).

    Returns None if scale is effectively zero (degenerate detection).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])  # (21, 3)

    wrist   = coords[0]
    mcp_mid = coords[9]   # middle-finger MCP — stable reference point

    scale = float(np.linalg.norm(mcp_mid - wrist))
    if scale < 1e-6:
        logger.debug("Degenerate hand detection (scale ~= 0); skipping hand.")
        return None

    normalized = (coords - wrist) / scale          # (21, 3)
    return normalized.flatten()                    # (63,)


def extract_feature_vector(result) -> np.ndarray:
    """
    Convert a HandLandmarker result into a fixed 126-dim feature vector.

    - Detected hands are sorted left-to-right by wrist x-coordinate for
      consistent ordering regardless of MediaPipe's internal labelling.
    - Missing hands (0 or 1 detected) are padded with zeros so the vector
      dimension is always exactly FEATURE_DIM.
    - If a hand is degenerate (normalization fails), its slot is zero-padded.

    Returns
    -------
    np.ndarray of shape (126,), dtype float32
    """
    feature_vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    if not result.hand_landmarks:
        return feature_vector

    # Sort hands left-to-right by wrist x-coordinate
    hands = sorted(result.hand_landmarks, key=lambda h: h[0].x)

    slot_offset = 0
    for hand in hands[:NUM_HANDS]:
        normalized = _normalize_hand(hand)
        if normalized is not None:
            feature_vector[slot_offset: slot_offset + 63] = normalized
        slot_offset += 63   # always advance slot even on failure -> consistent layout

    return feature_vector


# ──────────────────────────── Image Utilities ─────────────────────────────── #

def load_image_as_mp(img_path: str) -> Optional[Image]:
    """
    Read an image from disk and convert it to a MediaPipe Image.
    Applies mild CLAHE contrast enhancement to improve detection in
    dark / washed-out images.

    Returns None if the image cannot be read.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        logger.warning(f"Cannot read image: {img_path}")
        return None

    # CLAHE on L-channel for better detection in tricky lighting conditions
    lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab      = cv2.merge([clahe.apply(l), a, b])
    frame    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=rgb)


# ────────────────────────── Dataset Collection ────────────────────────────── #

def collect_dataset(detector: vision.HandLandmarker) -> list:
    """
    Walk DATASET_PATH, extract features for every image, and return raw rows.

    Directory structure expected:
        dataset/
            A/  image1.jpg  image2.jpg ...
            B/  image1.jpg ...
            ...
    """
    dataset_root = Path(DATASET_PATH)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: '{DATASET_PATH}'")

    labels = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    logger.info(f"Found {len(labels)} class labels: {labels}")

    rows            = []
    total_images    = 0
    skipped_no_hand = 0
    skipped_unread  = 0

    for label in labels:
        folder_path   = dataset_root / label
        image_files   = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        label_count   = 0
        label_skipped = 0

        for img_file in image_files:
            total_images += 1
            mp_image = load_image_as_mp(str(img_file))

            if mp_image is None:
                skipped_unread += 1
                continue

            try:
                result = detector.detect(mp_image)
            except Exception as exc:
                logger.warning(f"Detection failed for {img_file.name}: {exc}")
                skipped_unread += 1
                continue

            # Discard samples where NO hand was detected at all
            if not result.hand_landmarks:
                skipped_no_hand += 1
                label_skipped   += 1
                logger.debug(f"No hand detected: {img_file.name} [{label}]")
                continue

            feature_vector = extract_feature_vector(result)
            row = np.append(feature_vector, label)
            rows.append(row)
            label_count += 1
            logger.info(
                f"[{label}] Processed {img_file.name} "
                f"({len(result.hand_landmarks)} hand(s) detected)"
            )

        logger.info(
            f"Label '{label}': {label_count} saved, {label_skipped} skipped."
        )

    logger.info(
        f"\nSummary -> Total images: {total_images} | "
        f"Saved: {len(rows)} | "
        f"No-hand skipped: {skipped_no_hand} | "
        f"Unreadable: {skipped_unread}"
    )
    return rows


# ──────────────────────────── DataFrame Builder ───────────────────────────── #

def build_dataframe(rows: list) -> pd.DataFrame:
    """Assemble collected rows into a labelled, typed DataFrame."""
    feature_cols = [
        f"{i}_{c}"
        for i in range(NUM_HANDS * NUM_LANDMARKS)
        for c in ("x", "y", "z")
    ]
    columns = feature_cols + ["label"]

    df = pd.DataFrame(rows, columns=columns)

    # Cast feature columns to float32 for memory efficiency
    df[feature_cols] = df[feature_cols].astype(np.float32)

    return df


# ────────────────────────────── Entry Point ───────────────────────────────── #

def main() -> None:
    logger.info("=" * 60)
    logger.info("  ISL Two-Hand Landmark Extraction Pipeline")
    logger.info("=" * 60)

    detector = build_detector()
    logger.info(f"Detector ready  |  Model: {MODEL_PATH}")
    logger.info(f"Dataset path    |  {Path(DATASET_PATH).resolve()}")
    logger.info(
        f"Feature dim     |  {FEATURE_DIM}  "
        f"({NUM_HANDS} hands x {NUM_LANDMARKS} landmarks x 3 axes)"
    )

    rows = collect_dataset(detector)

    if not rows:
        logger.error(
            "No valid samples collected. "
            "Check your dataset path and image quality."
        )
        return

    df = build_dataframe(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"\n  Dataset saved  ->  '{OUTPUT_CSV}'")
    logger.info(f"   Shape         :  {df.shape}")
    logger.info(f"   Class dist.   :\n{df['label'].value_counts().to_string()}")
    logger.info(
        f"   Memory usage  :  "
        f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()