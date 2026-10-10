
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    multilabel_confusion_matrix,
)

DATA = Path("data/processed")
MODELS = Path("model_training/saved_models")
RESULTS = Path("model_training/evaluation_results")
RESULTS.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

X_val = val["comment_text"].fillna("")
y_val = val[LABELS].astype(int).to_numpy()

X_test = test["comment_text"].fillna("")
y_test = test[LABELS].astype(int).to_numpy()

all_results = {}

for name in ["logistic_regression", "linear_svm"]:
    path = MODELS / f"{name}.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    print(f"\nLoading {name}...", flush=True)
    model = joblib.load(path)

    # Obtain continuous scores for threshold tuning.
    classifier = model.named_steps["classifier"]
    tfidf = model.named_steps["tfidf"]

    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    scores_val = classifier.decision_function(X_val_tfidf)
    scores_test = classifier.decision_function(X_test_tfidf)

    best_threshold = 0.0
    best_f1 = -1.0
    tuning_rows = []

    # Tune only on validation data.
    for threshold in THRESHOLDS:
        pred_val = (scores_val >= threshold).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_val, pred_val, average="macro", zero_division=0
        )

        tuning_rows.append({
            "threshold": threshold,
            "macro_precision": float(p),
            "macro_recall": float(r),
            "macro_f1": float(f1),
            "subset_accuracy": float(accuracy_score(y_val, pred_val)),
        })

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = threshold

    pd.DataFrame(tuning_rows).to_csv(
        RESULTS / f"{name}_threshold_tuning.csv", index=False
    )

    # Evaluate test data once using the selected validation threshold.
    pred_test = (scores_test >= best_threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="macro", zero_division=0
    )

    per_label_p, per_label_r, per_label_f1, support = (
        precision_recall_fscore_support(
            y_test, pred_test, average=None, zero_division=0
        )
    )

    cm = multilabel_confusion_matrix(y_test, pred_test)

    result = {
        "model": name,
        "selected_threshold": best_threshold,
        "validation_macro_f1_at_selected_threshold": best_f1,
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_macro_f1": float(f1),
        "test_subset_accuracy": float(accuracy_score(y_test, pred_test)),
        "per_label": {},
        "per_label_confusion_matrices": {},
    }

    for i, label in enumerate(LABELS):
        result["per_label"][label] = {
            "precision": float(per_label_p[i]),
            "recall": float(per_label_r[i]),
            "f1": float(per_label_f1[i]),
            "support": int(support[i]),
        }
        result["per_label_confusion_matrices"][label] = (
            cm[i].tolist()
        )

    all_results[name] = result

    print(f"Selected validation threshold: {best_threshold}")
    print(f"Test macro precision: {p:.4f}")
    print(f"Test macro recall:    {r:.4f}")
    print(f"Test macro F1:        {f1:.4f}")
    print(f"Test subset accuracy: {result['test_subset_accuracy']:.4f}")
    print("Per-label test results:")
    for label, metrics in result["per_label"].items():
        print(
            f"  {label}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )

with open(RESULTS / "baseline_test_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved in: {RESULTS.resolve()}")
