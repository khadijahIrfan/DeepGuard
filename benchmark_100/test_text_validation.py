import sys
sys.path.insert(0, r'E:\DEEPGUARD')
from modules.text_analysis.text_analyzer import analyze_text_fake
import requests

print("=== LAYER 1: text_analyzer.py module tests ===")

# Test 1: Empty
r = analyze_text_fake('')
print("[1] Empty string    -> status:", r["status"], " prediction:", r["prediction"])

# Test 2: 10 chars (below 20 chars minimum)
r = analyze_text_fake('Short text')
print("[2] 10 chars        -> status:", r["status"], " msg:", r.get("message",""))

# Test 3: Exactly 19 chars (below minimum)
txt19 = '1234567890123456789'
r = analyze_text_fake(txt19)
print("[3] 19 chars        -> status:", r["status"], " char_count:", r.get("char_count",""))

# Test 4: Exactly 20 chars (should PASS to model)
txt20 = 'This has 20 chars!!!'
r = analyze_text_fake(txt20)
print("[4] 20 chars        -> status:", r["status"], " prediction:", r["prediction"], " fake_prob:", r.get("fake_probability"))

# Test 5: Normal news (well over 20 words)
news = 'Scientists at MIT have published a new study on climate change. The findings show sea levels rising by 3mm per year in coastal regions of South Asia according to satellite data.'
r = analyze_text_fake(news)
print("[5] Real news text  -> status:", r["status"], " prediction:", r["prediction"], " fake_prob:", r.get("fake_probability"), "%")

# Test 6: Over 1200 chars (should truncate and run)
long_text = ('climate change science news research report findings data analysis environmental ' * 25)
print("[6] Long text chars:", len(long_text), "->", end=" ")
r = analyze_text_fake(long_text)
print("status:", r["status"], " prediction:", r["prediction"])

print()
print("=== LAYER 2: app.py server-side route tests (via HTTP) ===")

# Test 7: Short text (< 20 words) via HTTP
r = requests.post('http://127.0.0.1:5000/predict',
    data={'text': 'This is too short.'},
    allow_redirects=False, timeout=5)
print("[7] HTTP 5 words    -> HTTP", r.status_code, " Body:", r.text[:150])

# Test 8: 19 words via HTTP
r = requests.post('http://127.0.0.1:5000/predict',
    data={'text': txt19},
    allow_redirects=False, timeout=5)
print("[8] HTTP 19 words   -> HTTP", r.status_code, " Body:", r.text[:150])

print()
print("All validation tests complete.")
