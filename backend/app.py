from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification
import torch

from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# =====================================
# Load DistilBERT Model
# =====================================

distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "deepguard_distilbert"
)

tokenizer = AutoTokenizer.from_pretrained(
    "deepguard_distilbert"
)

distilbert_model.eval()

# =====================================
# Sentiment Analyzer
# =====================================

analyzer = SentimentIntensityAnalyzer()

# =====================================
# Sentiment Function
# =====================================

def analyze_sentiment(text):

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive 😊"

    elif compound <= -0.05:
        return "Negative 😠"

    else:
        return "Neutral 😐"

# =====================================
# DistilBERT Prediction Function
# =====================================

def predict_distilbert(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = distilbert_model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    return prediction

# =====================================
# Routes
# =====================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


@app.route("/modules")
def modules():
    return render_template("modules.html")


@app.route("/about")
def about():
    return render_template("about.html")

# =====================================
# Prediction Route
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    # DistilBERT Prediction

    prediction = predict_distilbert(text)

    if prediction == 1:
        result = "FAKE 🔴"
    else:
        result = "REAL 🟢"

    # Sentiment

    sentiment = analyze_sentiment(text)

    return jsonify({
        "prediction": result,
        "sentiment": sentiment
    })

# =====================================
# Run Flask
# =====================================

if __name__ == "__main__":
    app.run(debug=True)