import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_classifier.preprocessing import transform_text


def test_lowercases_and_stems():
    assert transform_text("Winning PRIZES") == "win prize"


def test_removes_stopwords_and_punctuation():
    assert transform_text("This is a test!!!") == "test"


def test_empty_string():
    assert transform_text("") == ""
