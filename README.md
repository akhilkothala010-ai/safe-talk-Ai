# SafeTalk AI

**An AI-Based Cyberbullying and Toxic Comment Detection System**

SafeTalk AI aims to detect toxic and cyberbullying content in online comments using Natural Language Processing and Machine Learning. This project includes a user-friendly web interface powered by a pre-trained Transformer model (DistilBERT/BERT variant) to instantly classify text. It also includes comprehensive scripts to train your own models (Baseline ML & Transformers) on datasets like the Jigsaw Toxic Comment dataset.

## Features

- **User-Friendly Web Interface**: Built with Flask and Tailwind CSS.
- **Immediate Inference**: Uses Hugging Face's `pipeline` and `unitary/toxic-bert` for out-of-the-box toxicity detection without needing to train a model first.
- **Multi-Category Classification**: Detects standard toxicity, severe toxicity, obscenity, threats, insults, and identity hate.
- **Training Scripts Included**: 
  - `model_training/train_baseline.py`: Demonstrates TF-IDF vectorization with Logistic Regression, Naive Bayes, and SVM.
  - `model_training/train_transformer.py`: Demonstrates fine-tuning a DistilBERT model.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Create a virtual environment and install the required dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Web Application
To start the Flask server and use the web interface:

```bash
python app.py
```
Then, open your web browser and navigate to `http://127.0.0.1:5000/`.
*Note: The first time you run this, it will download the pre-trained model (~400MB) from Hugging Face.*

### 4. Running the Training Scripts
If you want to train the models yourself, navigate to the `model_training` directory. You will need to download the Jigsaw Toxic Comment dataset (from Kaggle) and point the scripts to your data.

```bash
python model_training/train_baseline.py
python model_training/train_transformer.py
```

---

## Study: Limitations and Bias in AI Moderation

An AI moderation system should not incorrectly classify normal communication as harmful. However, all NLP models have inherent limitations that must be addressed before deployment.

### 1. False Positives
A false positive occurs when the model flags a completely safe or benign comment as toxic.
* **Impact**: Users feel censored, leading to frustration and decreased engagement on the platform.
* **Cause**: The model might over-associate certain words (e.g., profanity used in a positive context like "That's crazy good!") with toxicity.

### 2. False Negatives
A false negative occurs when the model fails to detect actual cyberbullying or toxic content.
* **Impact**: Harmful content remains visible, damaging the community and violating platform safety guidelines.
* **Cause**: Malicious users often use obfuscation (e.g., "y0u are an id1ot"), heavy sarcasm, or nuanced threats that lack overt offensive keywords, which confuses the model.

### 3. Algorithmic Bias
Bias occurs when the model disproportionately flags language used by specific marginalized groups as toxic.
* **Impact**: Systemic discrimination against certain demographics (e.g., African American Vernacular English (AAVE) or LGBTQ+ terminology being incorrectly flagged).
* **Cause**: Imbalanced training data. If the annotators of the Jigsaw dataset harbored biases, or if the dataset contains historical prejudices, the model learns and amplifies those biases.

### 4. Lack of Context
Transformers (like BERT) are excellent at understanding surrounding words, but they still lack *situational* context.
* Two friends bantering playfully might use insults that the AI flags as severe toxicity. 
* The AI does not know the relationship between the users or the context of the broader conversation thread.

### Conclusion & Best Practices
To mitigate these issues, SafeTalk AI (and any moderation system) should ideally use a "Human-in-the-Loop" approach. The AI should flag highly suspicious content for human review rather than instantly banning users, and a mechanism for users to appeal false positive decisions must be implemented.
