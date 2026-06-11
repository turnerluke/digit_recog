from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py")


def _run():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_runs_without_exception():
    at = _run()
    assert not at.exception


def test_default_page_title_and_navigation():
    at = _run()
    assert at.title[0].value == "Drawn Digit Recognition"
    assert at.sidebar.selectbox[0].options == ["App", "Model", "Streamlit App"]


def test_model_page_renders():
    at = _run()
    at.sidebar.selectbox[0].set_value("Model").run()
    assert not at.exception
    text = " ".join(block.value for block in at.markdown)
    assert "Performance" in text
    assert "99.63%" in text


def test_streamlit_app_page_renders():
    at = _run()
    at.sidebar.selectbox[0].set_value("Streamlit App").run()
    assert not at.exception
    text = " ".join(block.value for block in at.markdown)
    assert "Streamlit" in text
