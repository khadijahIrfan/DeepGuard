import joblib

model = joblib.load("text_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text1 = "Scientists at a leading university have developed a new method for detecting early-stage cancer. The research was published in a peer-reviewed medical journal."

text2 = "Scientists have confirmed that drinking hot lemon water can make humans invisible for up to three hours."

vec1 = vectorizer.transform([text1])
vec2 = vectorizer.transform([text2])

print("Text 1:", model.predict(vec1))
print("Text 2:", model.predict(vec2))