"""Reproducibly train the LeNet-5 v2 digit classifier on MNIST.

Regenerates the served model artifact (``models/le_net5_v2.keras``), the
normalization constants used at inference (``models/normalization.json``), and a
record of the run's metrics (``models/metrics.json``).

Reproduce the full model:

    uv run python train.py

Quick smoke run (a couple epochs on a subset, throwaway output):

    uv run python train.py --epochs 1 --limit 2000 --output /tmp/smoke.keras
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from digit_recog.model import build_model

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_data(limit: int | None):
    """Load MNIST, pad 28x28 -> 32x32, and standardize with training statistics.

    Both splits are standardized with the *training* mean/std (not each split's
    own), which is the same transform applied at inference and avoids leaking
    test-set statistics. Returns the data plus those constants.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if limit:
        x_train, y_train = x_train[:limit], y_train[:limit]
        x_test, y_test = x_test[: limit // 5], y_test[: limit // 5]

    def pad(x):
        return np.pad(
            x.reshape(-1, 28, 28, 1), ((0, 0), (2, 2), (2, 2), (0, 0)), "constant"
        )

    x_train, x_test = pad(x_train), pad(x_test)
    mean, std = float(x_train.mean()), float(x_train.std())
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test), mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="Subset size for quick runs."
    )
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "le_net5_v2.keras")
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    (x_train, y_train), (x_test, y_test), mean, std = load_data(args.limit)

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    model = build_model()
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    # Only a full run produces valid serving constants; a subset run would
    # compute statistics over partial data, so leave the committed file intact.
    if not args.limit:
        norm_path = MODELS_DIR / "normalization.json"
        norm_path.write_text(
            json.dumps(
                {
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "_source": "MNIST training set, reshaped to 28x28, padded 2px to 32x32, global mean/std.",
                },
                indent=2,
            )
            + "\n"
        )

    metrics = {
        "epochs": args.epochs,
        "seed": args.seed,
        "tensorflow": tf.__version__,
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
        "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved model to {args.output}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
