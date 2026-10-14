from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = r"E:\DeepGuard\models\final text"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")

text = """
The government has announced a new education policy to improve
the quality of schools and provide better learning opportunities
for students across the country.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=1)

prediction = torch.argmax(probabilities, dim=1).item()

if prediction == 0:
    label = "REAL"
else:
    label = "FAKE"

confidence = probabilities[0][prediction].item() * 100

print()
print("========== DeepGuard Prediction ==========")
print("Prediction:", label)
print("Confidence:", f"{confidence:.2f}%")
print("==========================================")