# SMS Spam Classifier (NLP)

Classifies an SMS / email message as **Spam** or **Not Spam** using NLTK text preprocessing, TF-IDF features and a Multinomial Naive Bayes model, served through a Streamlit web app.

## Project structure

```
sms-spam-classifier/
├── app/
│   └── streamlit_app.py      # Streamlit web UI
├── src/spam_classifier/
│   ├── config.py             # paths and constants
│   ├── preprocessing.py      # transform_text (shared by training + app)
│   └── model.py              # SpamClassifier: load artifacts & predict
├── scripts/
│   └── train.py              # retrain and save models/
├── notebooks/
│   └── classification.ipynb  # EDA and experiments
├── data/raw/spam.csv         # dataset
├── models/                   # model.pkl, vectorizer.pkl
├── tests/                    # pytest tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NLTK data (`punkt`, `stopwords`) is downloaded automatically on first run.

## Usage

Run the app:

```bash
streamlit run app/streamlit_app.py
```

Retrain the model:

```bash
python scripts/train.py
```

Run tests:

```bash
pytest
```
