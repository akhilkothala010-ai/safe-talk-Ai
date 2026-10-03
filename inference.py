import os
# Suppress huggingface warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline

class ToxicClassifier:
    def __init__(self):
        """
        Initializes the text classification pipeline.
        We use 'unitary/toxic-bert', a model trained on the Jigsaw Toxic Comment dataset.
        It can classify text into multiple categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.
        """
        print("Loading pre-trained toxic comment model. This may take a moment...")
        self.classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert", 
            top_k=None # Get scores for all labels
        )
        print("Model loaded successfully.")

    def predict(self, text):
        """
        Predicts toxicity of the input text.
        """
        if not text or not text.strip():
            return {"is_toxic": False, "predictions": {}}

        # Get pipeline predictions
        results = self.classifier(text)
        
        # pipeline with top_k=None returns a list of dictionaries for each input
        # e.g., [[{'label': 'toxic', 'score': 0.9}, {'label': 'obscene', 'score': 0.1}, ...]]
        
        predictions = {}
        is_toxic = False
        
        # Sort results by score in descending order
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        
        for item in sorted_results:
            label = item['label']
            score = round(item['score'], 4)
            predictions[label] = score
            
            # If any harmful category is detected with > 0.5 confidence, flag as toxic
            if score > 0.5:
                is_toxic = True
                
        return {
            "is_toxic": is_toxic,
            "predictions": predictions
        }

# For testing
if __name__ == "__main__":
    clf = ToxicClassifier()
    print("Test 1 (Normal):", clf.predict("Hello, how are you doing today?"))
    print("Test 2 (Toxic):", clf.predict("You are an absolute idiot and I hate you!"))
