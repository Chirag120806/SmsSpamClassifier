"""Load the trained artifacts and run predictions."""
import pickle
from pathlib import Path

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocessing import transform_text


class SpamClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH):
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, message: str) -> int:
        """Return 1 for spam, 0 for not spam."""
        vector = self.vectorizer.transform([transform_text(message)])
        return int(self.model.predict(vector)[0])

    def is_spam(self, message: str) -> bool:
        return self.predict(message) == 1
