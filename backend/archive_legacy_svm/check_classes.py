import joblib

model = joblib.load("text_model.pkl")

print(model.classes_)