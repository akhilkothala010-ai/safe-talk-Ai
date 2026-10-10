
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load the original Jigsaw dataset
df = pd.read_csv("data/extracted/train.csv")

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Keep all toxic comments and sample clean comments
df["any_toxic"] = df[label_cols].max(axis=1)

toxic = df[df["any_toxic"] == 1]
clean = df[df["any_toxic"] == 0]

# CPU-friendly sample: all toxic comments + 25,000 clean comments
clean_sample = clean.sample(n=25000, random_state=42)
sample = pd.concat([toxic, clean_sample], ignore_index=True)
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

# Fixed train/validation/test split: 70% / 15% / 15%
train, temp = train_test_split(
    sample,
    test_size=0.30,
    random_state=42,
    stratify=sample["any_toxic"],
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["any_toxic"],
)

# Save splits
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

train.to_csv(out / "train.csv", index=False)
val.to_csv(out / "validation.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print("Dataset preparation complete")
print("Sample size:", len(sample))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("\nAny-toxicity distribution:")
for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(
        f"{name}: clean={(part['any_toxic'] == 0).sum()}, "
        f"toxic={(part['any_toxic'] == 1).sum()}"
    )

print("\nSix-label positive counts in full sample:")
print(sample[label_cols].sum().to_string())
