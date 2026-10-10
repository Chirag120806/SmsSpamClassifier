"""Text preprocessing shared by training and the web app."""
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_stemmer = PorterStemmer()


def ensure_nltk_data() -> None:
    """Download required NLTK resources if they are missing."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    for package, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> frozenset:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


def transform_text(text: str) -> str:
    """Lowercase, tokenize, drop non-alphanumerics/stopwords, then stem."""
    ensure_nltk_data()
    stop_words = _stop_words()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(_stemmer.stem(t) for t in tokens)
