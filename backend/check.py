import joblib

model = joblib.load("text_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

texts = [
    "The and is are was were",
    "Breaking news government announces new policy",
    "Aliens landed in Lahore and gave free gold to everyone"
]

X = vectorizer.transform(texts)

print(model.predict(X))