import torch
import shap

from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification
)


# =====================================
# Load DistilBERT Model
# =====================================

MODEL_PATH = r"E:\DeepGuard\models\final_distilbert_2026\FINAL_MODEL"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# =====================================
# Prediction Function
# =====================================

def predict_proba(texts):

    # Convert SHAP input to a normal Python list
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    # Make sure every item is a Python string
    texts = [
        str(text)
        for text in texts
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    return probabilities.detach().cpu().numpy()


# =====================================
# SHAP Explainer
# =====================================

explainer = shap.Explainer(
    predict_proba,
    shap.maskers.Text(tokenizer)
)


# =====================================
# Explain Prediction
# =====================================

def explain_prediction(text):

    shap_values = explainer(
        [text],
        max_evals=100
    )

    return shap_values


def get_important_words(text, top_k=5):

    # Get SHAP explanation
    shap_values = explain_prediction(text)

    # Get model prediction
    probabilities = predict_proba([text])

    prediction = int(
        probabilities[0].argmax()
    )

    # Get tokens
    tokens = shap_values.data[0]

    # Get SHAP values
    values = shap_values.values[0]

    important_words = []

    for token, value in zip(tokens, values):

        token = token.strip()

        # Skip empty tokens
        if not token:
            continue

        # Get contribution for predicted class
        contribution = float(
            value[prediction]
        )

        # Determine direction
        if contribution > 0:

            direction = (
                "Supports "
                + (
                    "FAKE"
                    if prediction == 1
                    else "REAL"
                )
            )

        else:

            direction = (
                "Opposes "
                + (
                    "FAKE"
                    if prediction == 1
                    else "REAL"
                )
            )

        important_words.append({

            "word": token,

            "contribution": round(
                contribution,
                4
            ),

            "direction": direction

        })


    # Strongest contributions first
    important_words.sort(
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    return important_words[:top_k]


    return important_words[:top_k]