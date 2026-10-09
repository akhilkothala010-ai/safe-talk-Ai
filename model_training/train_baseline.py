"""
SafeTalk AI - Baseline Machine Learning Models

This script demonstrates how to train traditional ML models (Logistic Regression,
Naive Bayes, SVM) on a dataset for toxic comment classification.
It uses TF-IDF for feature extraction.

Dataset expected format: A CSV file with a 'comment_text' column and a 'toxic' label column.
For demonstration, this script uses a small synthetic dataset if no real dataset is provided.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation/numbers, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove noise (punctuation, special chars, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def load_dataset():
    """Loads the Jigsaw Toxic Comment dataset."""
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
        'toxic': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def train_and_evaluate(df):
    """Trains and evaluates baseline models."""
    print("Preprocessing text...")
    df['clean_text'] = df['comment_text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['toxic']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    df = load_dataset()
    train_and_evaluate(df)
