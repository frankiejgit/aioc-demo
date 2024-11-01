from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Configuration (load from environment variables)
MODEL_DIR = "trained_model/" # Path to the trained model in Cloud Storage

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

    # Map class ID to sentiment (0: negative, 1: positive)
    sentiment = "positive" if predicted_class_id == 1 else "negative"

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))