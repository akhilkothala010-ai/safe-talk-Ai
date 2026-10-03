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

def get_dummy_dataset():
    """Returns a synthetic dataset for demonstration."""
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
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Hugging Face Trainer expects 'label' column
    }
    return pd.DataFrame(data)

def compute_metrics(pred):
    """Computes accuracy and f1 score for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def train_transformer():
    print("Loading dataset...")
    # For real use, load your dataset here:
    # df = pd.read_csv('path_to_data.csv')
    df = get_dummy_dataset()
    
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
