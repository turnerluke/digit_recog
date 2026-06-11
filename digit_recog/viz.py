import altair as alt
import numpy as np
import pandas as pd


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
