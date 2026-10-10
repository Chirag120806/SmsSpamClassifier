"""Train the model and save artifacts to models/.   Usage: python scripts/train.py"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from spam_classifier import config
from spam_classifier.preprocessing import transform_text


def main() -> None:
    df = pd.read_csv(config.DATA_PATH, encoding="latin2")[["v1", "v2"]]
    df.columns = ["target", "text"]
    df["target"] = (df["target"] == "spam").astype(int)
    df = df.drop_duplicates(keep="first")
    df["transformed_text"] = df["text"].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=config.MAX_FEATURES)
    X = tfidf.fit_transform(df["transformed_text"])
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(config.VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved artifacts to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
