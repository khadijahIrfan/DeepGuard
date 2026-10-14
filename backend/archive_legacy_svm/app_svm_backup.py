from flask import Flask, request, jsonify, render_template
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load model and vectorizer

model = joblib.load("text_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sentiment analyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):


    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive 😊"

    elif compound <= -0.05:
        return "Negative 😠"

    else:
        return "Neutral 😐"


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

@app.route("/predict", methods=["POST"])
def predict():


    data = request.get_json()

    text = data["text"]

# Text Detection
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        result = "FAKE 🔴"
    else:
        result = "REAL 🟢"

# Sentiment Analysis
    sentiment = analyze_sentiment(text)

    print("SENTIMENT =", sentiment)

    return jsonify({
    "prediction": result,
    "sentiment": sentiment
})


if __name__ == "__main__":
    app.run(debug=True)
