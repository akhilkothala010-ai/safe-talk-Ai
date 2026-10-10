
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    multilabel_confusion_matrix,
)

DATA = Path("data/processed")
OUT = Path("model_training/evaluation_results")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

MODEL_NAME = "unitary/toxic-bert"
BATCH_SIZE = 16


test = pd.read_csv(DATA / "test.csv")
test = test.sample(
    n=min(2000, len(test)),
    random_state=42
).reset_index(drop=True)

texts = test["comment_text"].fillna("").astype(str).tolist()
y_test = test[LABELS].astype(int).to_numpy()

print(f"Test comments: {len(texts)}", flush=True)
print(f"Loading {MODEL_NAME}...", flush=True)

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,
    device=-1,
)
print("Model labels:", classifier.model.config.id2label)

scores = []

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]
    outputs = classifier(
        batch,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=512,
    )

    for item in outputs:
        # Transformers may return a list of label-score dictionaries.
        if item and isinstance(item[0], dict):
            label_scores = {x["label"].lower(): x["score"] for x in item}
        else:
            label_scores = {x["label"].lower(): x["score"] for x in item}

        scores.append([
            label_scores.get(label, 0.0)
            for label in LABELS
        ])

    done = min(start + BATCH_SIZE, len(texts))
    if done % 256 == 0 or done == len(texts):
        print(f"Scored {done}/{len(texts)} comments", flush=True)

scores = np.asarray(scores, dtype=float)

# Select a threshold using validation data only.
# Use the same validation split and the same pretrained model.

val = pd.read_csv(DATA / "validation.csv")
val = val.sample(
    n=min(2000, len(val)),
    random_state=42
).reset_index(drop=True)

val_texts = val["comment_text"].fillna("").astype(str).tolist()
y_val = val[LABELS].astype(int).to_numpy()

val_scores = []
for start in range(0, len(val_texts), BATCH_SIZE):
    batch = val_texts[start:start + BATCH_SIZE]
    outputs = classifier(
        batch,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=512,
    )

    for item in outputs:
        label_scores = {x["label"].lower(): x["score"] for x in item}
        val_scores.append([
            label_scores.get(label, 0.0)
            for label in LABELS
        ])

val_scores = np.asarray(val_scores, dtype=float)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
tuning = []
best_threshold = 0.5
best_f1 = -1.0

for threshold in thresholds:
    pred_val = (val_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, pred_val, average="macro", zero_division=0
    )
    tuning.append({
        "threshold": threshold,
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "subset_accuracy": float(accuracy_score(y_val, pred_val)),
    })
    if f1 > best_f1:
        best_f1 = float(f1)
        best_threshold = threshold

pd.DataFrame(tuning).to_csv(
    OUT / "toxic_bert_threshold_tuning.csv", index=False
)

pred_test = (scores >= best_threshold).astype(int)
p, r, f1, _ = precision_recall_fscore_support(
    y_test, pred_test, average="macro", zero_division=0
)
per_p, per_r, per_f1, support = precision_recall_fscore_support(
    y_test, pred_test, average=None, zero_division=0
)
cms = multilabel_confusion_matrix(y_test, pred_test)

result = {
    "model": MODEL_NAME,
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
        "precision": float(per_p[i]),
        "recall": float(per_r[i]),
        "f1": float(per_f1[i]),
        "support": int(support[i]),
    }
    result["per_label_confusion_matrices"][label] = cms[i].tolist()

with open(OUT / "toxic_bert_test_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\nToxic-BERT test results")
print(f"Selected validation threshold: {best_threshold}")
print(f"Test macro precision: {p:.4f}")
print(f"Test macro recall:    {r:.4f}")
print(f"Test macro F1:        {f1:.4f}")
print(f"Test subset accuracy: {result['test_subset_accuracy']:.4f}")

for label, metrics in result["per_label"].items():
    print(
        f"{label}: P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"support={metrics['support']}"
    )

print(f"\nSaved results to: {OUT.resolve()}")
