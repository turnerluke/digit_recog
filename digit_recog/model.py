from pathlib import Path

import keras
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.regularizers import l2

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "le_net5_v2.keras"
)


def build_model() -> Sequential:
    """Modified LeNet-5 (v2). Layer numbering counts only learnable layers."""
    model = Sequential(
        [
            Conv2D(
                32,
                5,
                strides=1,
                activation="relu",
                input_shape=(32, 32, 1),
                kernel_regularizer=l2(0.0005),
                name="convolution_1",
            ),
            Conv2D(32, 5, strides=1, use_bias=False, name="convolution_2"),
            BatchNormalization(name="batchnorm_1"),
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2, name="max_pool_1"),
            Dropout(0.25, name="dropout_1"),
            Conv2D(
                64,
                3,
                strides=1,
                activation="relu",
                kernel_regularizer=l2(0.0005),
                name="convolution_3",
            ),
            Conv2D(64, 3, strides=1, use_bias=False, name="convolution_4"),
            BatchNormalization(name="batchnorm_2"),
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2, name="max_pool_2"),
            Dropout(0.25, name="dropout_2"),
            Flatten(name="flatten"),
            Dense(256, use_bias=False, name="fully_connected_1"),
            BatchNormalization(name="batchnorm_3"),
            Activation("relu"),
            Dense(128, use_bias=False, name="fully_connected_2"),
            BatchNormalization(name="batchnorm_4"),
            Activation("relu"),
            Dense(84, use_bias=False, name="fully_connected_3"),
            BatchNormalization(name="batchnorm_5"),
            Activation("relu"),
            Dropout(0.25, name="dropout_3"),
            Dense(10, activation="softmax", name="output"),
        ]
    )
    model._name = "LeNet5v2"
    return model


def load_model(path: Path = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load the trained model from disk (path resolved relative to the repo)."""
    return keras.models.load_model(path)
