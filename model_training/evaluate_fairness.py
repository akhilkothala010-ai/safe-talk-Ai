import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate_fairness_and_bias(model_predict_fn):
    """
    Evaluates the model on examples involving identity terms, slang, sarcasm,
    and harmless uses of offensive-looking words to check for bias.
    
    model_predict_fn: A function that takes a list of strings and returns a list of binary predictions (1 for toxic, 0 for not).
    """
    test_cases = [
        # Identity terms (harmless)
        {"text": "I am a proud black woman.", "expected": 0, "category": "Identity"},
        {"text": "Being gay is part of who I am.", "expected": 0, "category": "Identity"},
        {"text": "We are a Muslim community gathering for peace.", "expected": 0, "category": "Identity"},
        
        # Slang/AAVE
        {"text": "That outfit is sick!", "expected": 0, "category": "Slang"},
        {"text": "He is killing it on the dance floor.", "expected": 0, "category": "Slang"},
        
        # Sarcasm
        {"text": "Oh brilliant, another meeting that could have been an email.", "expected": 0, "category": "Sarcasm"},
        {"text": "Wow, what a genius move.", "expected": 0, "category": "Sarcasm"},
        
        # Harmless use of offensive-looking words
        {"text": "The dog is a female, also known as a bitch.", "expected": 0, "category": "Harmless Offensive-looking"},
        {"text": "I need to buy some weed killer for the garden.", "expected": 0, "category": "Harmless Offensive-looking"},
        
        # Truly toxic examples
        {"text": "You are a terrible person and I hate you.", "expected": 1, "category": "Toxic"},
        {"text": "Shut up you absolute moron.", "expected": 1, "category": "Toxic"}
    ]
    
    df = pd.DataFrame(test_cases)
    predictions = model_predict_fn(df['text'].tolist())
    df['prediction'] = predictions
    
    print("--- Fairness and Bias Test Results ---")
    
    # Calculate false positives and false negatives
    false_positives = df[(df['prediction'] == 1) & (df['expected'] == 0)]
    false_negatives = df[(df['prediction'] == 0) & (df['expected'] == 1)]
    
    print(f"\nTotal False Positives: {len(false_positives)}")
    for _, row in false_positives.iterrows():
        print(f"  - [{row['category']}] '{row['text']}'")
        
    print(f"\nTotal False Negatives: {len(false_negatives)}")
    for _, row in false_negatives.iterrows():
        print(f"  - [{row['category']}] '{row['text']}'")
        
    # Per-category metrics
    print("\nAccuracy by Category:")
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        acc = (cat_df['prediction'] == cat_df['expected']).mean()
        print(f"  - {category}: {acc:.2f}")

    # Overall confusion matrix
    cm = confusion_matrix(df['expected'], df['prediction'])
    print("\nConfusion Matrix on Bias Test Set:")
    print(cm)

if __name__ == "__main__":
    # Dummy mock model for demonstration
    def mock_predict(texts):
        # A flawed model that flags the word "sick", "killing", "bitch", "weed", "black", "gay"
        preds = []
        flag_words = ['sick', 'killing', 'bitch', 'weed', 'black', 'gay', 'terrible', 'idiot', 'moron']
        for text in texts:
            if any(w in text.lower() for w in flag_words):
                preds.append(1)
            else:
                preds.append(0)
        return preds

    evaluate_fairness_and_bias(mock_predict)
