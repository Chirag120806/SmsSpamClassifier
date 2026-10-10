"""SMS / Email spam classifier."""
from .preprocessing import transform_text
from .model import SpamClassifier

__all__ = ["transform_text", "SpamClassifier"]
