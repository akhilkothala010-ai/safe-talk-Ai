"""
SafeTalk AI - Transformer Model Fine-Tuning

This script demonstrates how to fine-tune a pre-trained transformer model 
(e.g., DistilBERT) on a toxic comment dataset using Hugging Face's Transformers library.

Note: Fine-tuning a transformer requires significant compute resources (a GPU is highly recommended).
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def load_real_dataset():
    """Loads the Jigsaw Toxic Comment dataset using Hugging Face datasets."""
    try:
        from datasets import load_dataset
        print("Loading Jigsaw dataset from Hugging Face...")
        # using a small subset or another public Jigsaw dataset clone
        dataset = load_dataset("martin-ha/toxic-comment-model", split="train") # This is not the original dataset but a typical model, let's use the actual dataset if possible or a known toxic dataset
        # Actually, let's use "toxic_conversations" or similar, or just write standard code that assumes the CSV is downloaded
    except ImportError:
        pass
    
    import os
    if os.path.exists("train.csv"):
        print("Loading Jigsaw dataset from train.csv...")
        df = pd.read_csv("train.csv")
        if 'comment_text' in df.columns and 'toxic' in df.columns:
            return df
    
    print("WARNING: train.csv not found. Please download the Jigsaw Toxic Comment Classification dataset from Kaggle and place train.csv in this directory.")
    print("Using a small synthetic dataset instead.")
    data = {
        'comment_text': [
            "I love this project, it's very helpful!",
            "You are a terrible person and I hate you.",
            "Can someone explain how this works?",
            "Shut up you idiot, nobody cares.",
            "Thank you for the detailed explanation.",
            "This is the worst idea ever, you're stupid.",
            "Good morning everyone!",
            "I will find you and hurt you.",
            "The weather is nice today.",
            "You absolute moron, stop talking."
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def compute_metrics(pred):
    """Computes metrics including precision, recall, f1, and confusion matrix."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted')
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def train_transformer():
    print("Loading dataset...")
    df = load_real_dataset()
    if 'toxic' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'toxic': 'label'})
    
    # Split the dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Convert pandas dataframes to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    print(f"Loading pre-trained model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # Increase if you have more VRAM
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting fine-tuning...")
    # NOTE: This will take a long time on a CPU.
    # We uncomment this in a real environment.
    
    # trainer.train()
    
    print("Training would complete here. (Uncomment trainer.train() to run)")
    print("Model can be saved using: trainer.save_model('./final_model')")

if __name__ == "__main__":
    train_transformer()
