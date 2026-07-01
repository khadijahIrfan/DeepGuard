from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("text_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "DeepGuard Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        result = "FAKE 🔴"
    else:
        result = "REAL 🟢"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)