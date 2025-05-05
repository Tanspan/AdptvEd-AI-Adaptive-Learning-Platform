from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import os
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure model cache
os.environ['TRANSFORMERS_CACHE'] = './model_cache'
os.makedirs('./model_cache', exist_ok=True)

# Initialize text predictor
try:
    print("Loading DistilGPT-2 model...")
    text_predictor = pipeline(
        "text-generation",
        model="distilgpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    text_predictor = None


@app.route('/')
def home():
    return render_template('notes.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not text_predictor:
        return jsonify({"error": "Prediction model not available"}), 503

    text = request.json.get('text', '')
    if not text.strip():
        return jsonify({"prediction": ""})

    try:
        # Generate prediction with temperature for variability
        prediction = text_predictor(
            text,
            max_length=len(text.split()) + 5,  # Predict 5 more tokens
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )[0]['generated_text']

        # Extract only new words
        new_text = prediction[len(text):].strip()
        next_words = ' '.join(new_text.split()[:3])  # Get next 3 words

        return jsonify({
            "prediction": next_words,
            "full_prediction": prediction  # For debugging
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)