"""Central place for paths and constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "spam.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

MAX_FEATURES = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 2
