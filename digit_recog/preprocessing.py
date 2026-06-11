import json
from pathlib import Path

import cv2
import numpy as np

# Normalization constants from the training set (see models/normalization.json).
# Serving must standardize with these fixed values, not per-image statistics, to
# match how the model was trained and avoid train/serve skew.
_NORM = json.loads(
    (
        Path(__file__).resolve().parent.parent / "models" / "normalization.json"
    ).read_text()
)
TRAIN_MEAN: float = _NORM["mean"]
TRAIN_STD: float = _NORM["std"]

_BLANK_INPUT = np.zeros((1, 32, 32, 1))


def _center_by_mass(img: np.ndarray) -> np.ndarray:
    """Shift a single-channel image so its center of mass sits at the center,
    mirroring how MNIST digits are centered."""
    total = img.sum()
    if total == 0:
        return img
    rows, cols = np.indices(img.shape)
    cy, cx = (rows * img).sum() / total, (cols * img).sum() / total
    shift_y = int(round(img.shape[0] / 2 - cy))
    shift_x = int(round(img.shape[1] / 2 - cx))
    return np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Convert a raw RGBA drawing canvas into the (1, 32, 32, 1) tensor the model
    expects, reproducing the MNIST preprocessing the model was trained on:
    isolate the digit, size-normalize it into a 20px box centered in 28x28 by
    center of mass, pad to 32x32, and standardize with the training statistics.
    """
    gray = np.mean(img[..., :3], axis=2).astype(np.float32)
    ink = 255.0 - gray  # MNIST polarity: bright ink on a dark background

    # Flatten the (near-uniform) background to zero, then bail on a blank canvas.
    ink = np.clip(ink - float(np.median(ink)), 0, None)
    if ink.max() < 1e-6:
        return _BLANK_INPUT.copy()

    # Crop to the digit's bounding box.
    mask = ink > ink.max() * 0.15
    rows, cols = np.where(mask)
    crop = ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    # Size-normalize the longest side to 20px, preserving aspect ratio.
    height, width = crop.shape
    scale = 20.0 / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place into a 28x28 frame, recenter by mass, pad to 32x32.
    framed = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - new_h) // 2, (28 - new_w) // 2
    framed[y0 : y0 + new_h, x0 : x0 + new_w] = crop
    framed = _center_by_mass(framed)
    padded = np.pad(framed, ((2, 2), (2, 2)), "constant")

    standardized = (padded - TRAIN_MEAN) / TRAIN_STD
    return standardized[np.newaxis, ..., np.newaxis]
