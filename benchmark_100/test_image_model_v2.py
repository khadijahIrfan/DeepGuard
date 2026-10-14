import sys
import glob
from pathlib import Path
sys.path.insert(0, r"E:\DEEPGUARD")
from modules.image_detection import predict_image_authenticity

real_images = glob.glob(r"C:\Users\GA LAPTOP\Downloads\WhatsApp Image *.jpeg")[:5]
fake_images = [
    r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\ai_generated_fake_test_1_1791655119454.jpg",
    r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\ai_generated_fake_test_2_1791655132210.jpg"
]

print("=== TESTING REAL IMAGES (WhatsApp) ===")
real_correct = 0
for img in real_images:
    res = predict_image_authenticity(img)
    pred = res.get('prediction', 'ERROR')
    print(f"File: {Path(img).name} -> Predicted: {pred} (FAKE prob: {res.get('fake_probability',0):.2f})")
    if pred == "REAL": real_correct += 1

print(f"\nREAL Images Accuracy: {real_correct} / {len(real_images)}")

print("\n=== TESTING FAKE IMAGES (AI Generated) ===")
fake_correct = 0
for img in fake_images:
    res = predict_image_authenticity(img)
    pred = res.get('prediction', 'ERROR')
    print(f"File: {Path(img).name} -> Predicted: {pred} (FAKE prob: {res.get('fake_probability',0):.2f})")
    if pred == "FAKE": fake_correct += 1

print(f"\nFAKE Images Accuracy: {fake_correct} / {len(fake_images)}")
