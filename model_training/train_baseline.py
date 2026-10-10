
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

DATA = Path("data/processed")
OUT = Path("model_training/saved_models")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")

X_train = train["comment_text"].fillna("")
y_train = train[LABELS].astype(int)
X_val = val["comment_text"].fillna("")
y_val = val[LABELS].astype(int)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "linear_svm": LinearSVC(
        class_weight="balanced",
        dual="auto",
    ),
}

for name, classifier in models.items():
    print(f"\nTraining {name}...", flush=True)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", OneVsRestClassifier(classifier)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    subset_accuracy = accuracy_score(y_val, predictions)

    print(f"Validation macro precision: {precision:.4f}")
    print(f"Validation macro recall:    {recall:.4f}")
    print(f"Validation macro F1:        {f1:.4f}")
    print(f"Validation subset accuracy: {subset_accuracy:.4f}")

    path = OUT / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model: {path}")

print("\nBaseline training complete.")
