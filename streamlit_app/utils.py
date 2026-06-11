import numpy as np
import cv2
import altair as alt
import pandas as pd


def preprocess_image(img: np.ndarray) -> np.ndarray:

    img = np.mean(img, axis=2)  # To B&W image (2 dimensional)
    img = cv2.resize(
        img, (32, 32), interpolation=cv2.INTER_AREA
    )  # To (32, 32) for model input

    # Scale vals from 0 -> 1, guarding against a uniform (blank) canvas
    value_range = img.max() - img.min()
    if value_range == 0:
        return np.zeros((1, 32, 32, 1))
    img = (img - img.min()) / value_range
    # Inverse vals (vals = 1 - vals)
    img = 1 - img
    # Standard scale vals (vals = (vals - vals_mean) / std_vals
    std = img.std()
    if std > 0:
        img = (img - img.mean()) / std

    img = np.expand_dims(np.expand_dims(img, 0), 3)  # Reshape to (None, 32, 32, None)
    return img


def create_certainty_chart(output: np.ndarray) -> alt.Chart:
    """
    Builds an Altair bar chart of the model's per-digit confidence, with the
    predicted digit highlighted.
    """
    confidence = output[0] * 100
    predicted = int(confidence.argmax())

    data = pd.DataFrame(
        {
            "digit": [str(i) for i in range(10)],
            "confidence": confidence,
            "predicted": [i == predicted for i in range(10)],
        }
    )

    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("digit:N", title="Digit", sort=None),
            y=alt.Y(
                "confidence:Q",
                title="Confidence (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.condition(
                alt.datum.predicted,
                alt.value("#ff4b4b"),  # Streamlit accent for the prediction
                alt.value("#83c9ff"),  # muted blue for the rest
            ),
            tooltip=[
                alt.Tooltip("digit:N", title="Digit"),
                alt.Tooltip("confidence:Q", title="Confidence (%)", format=".2f"),
            ],
        )
    )
