from flask import Flask, render_template, request, jsonify
from inference import ToxicClassifier

app = Flask(__name__)

# Initialize the classifier when the app starts
# This will download the model on the first run if it's not cached locally
print("Initializing SafeTalk AI Application...")
classifier = ToxicClassifier()

@app.route('/')
def home():
    """Renders the main web interface."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the About page."""
    return render_template('about.html')

@app.route('/team')
def team():
    """Renders the Team page."""
    return render_template('team.html')

@app.route('/evaluation')
def evaluation():
    """Renders the Evaluation and Metrics page."""
    return render_template('evaluation.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to analyze comments.
    Expects JSON: {"text": "the comment to analyze"}
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text']
    
    try:
        # Perform prediction
        result = classifier.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    API endpoint to record user feedback when they flag a prediction as incorrect.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    # Append to a feedback file
    try:
        import json
        with open('feedback.jsonl', 'a') as f:
            f.write(json.dumps(data) + '\n')
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app in debug mode but disable auto-reloader 
    # to avoid loading the heavy AI model twice in memory
    app.run(debug=True, use_reloader=False, port=5000)
