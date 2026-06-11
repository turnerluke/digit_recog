from pathlib import Path

import cv2
import numpy as np
import pytest

from digit_recog.model import build_model, load_model
from digit_recog.preprocessing import preprocess_image
from digit_recog.viz import create_certainty_chart

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("digit_*.png"))


@pytest.fixture(scope="session")
def model():
    return load_model()


def test_build_model_has_expected_io_shape():
    net = build_model()
    assert net.input_shape == (None, 32, 32, 1)
    assert net.output_shape == (None, 10)


def _canvas(fill_blob: bool) -> np.ndarray:
    """A 300x300 RGBA canvas, optionally with a dark blob drawn on it."""
    img = np.full((300, 300, 4), 230, dtype=np.uint8)
    if fill_blob:
        img[100:200, 130:170, :3] = 0
    return img


def test_preprocess_returns_model_input_shape():
    out = preprocess_image(_canvas(fill_blob=True))
    assert out.shape == (1, 32, 32, 1)


def test_preprocess_blank_canvas_is_safe():
    """A uniform canvas must not divide by zero or produce NaNs."""
    out = preprocess_image(_canvas(fill_blob=False))
    assert out.shape == (1, 32, 32, 1)
    assert not np.isnan(out).any()


def test_model_outputs_probability_distribution(model):
    out = model.predict(preprocess_image(_canvas(fill_blob=True)), verbose=0)
    assert out.shape == (1, 10)
    assert out.sum() == pytest.approx(1.0, abs=1e-4)
    assert 0 <= int(out.argmax()) <= 9


def test_certainty_chart_has_ten_bars():
    output = np.array([[0.01, 0.02, 0.9, 0.0, 0.01, 0.0, 0.03, 0.0, 0.02, 0.01]])
    spec = create_certainty_chart(output).to_dict()
    (dataset,) = spec["datasets"].values()
    assert spec["mark"]["type"] == "bar"
    assert len(dataset) == 10


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_classifies_canvas_fixtures(model, fixture):
    """End-to-end golden test: digits drawn at varied sizes/positions must be
    classified correctly. Guards against train/serve preprocessing skew."""
    expected = int(fixture.stem.split("_")[1])
    rgba = cv2.imread(str(fixture), cv2.IMREAD_UNCHANGED)
    prediction = int(model.predict(preprocess_image(rgba), verbose=0).argmax())
    assert prediction == expected
