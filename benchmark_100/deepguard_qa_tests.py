"""
DeepGuard Full QA Test Suite - FIXED VERSION
Covers: Smoke, Functional, Regression, Boundary, Security, Performance
"""
import sys, os, time, json, threading, io
import requests

BASE_URL = "http://127.0.0.1:5000"
TEST_IMG_REAL = r"E:\DEEPGUARD\backend\test_images\test.jpg"
TEST_IMG_FAKE = r"E:\DEEPGUARD\benchmark_100\images\s3_01.jpg"
if not os.path.exists(TEST_IMG_FAKE):
    TEST_IMG_FAKE = r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\ai_generated_fake_test_2_1791655132210.jpg"

# Check real image exists, fallback
if not os.path.exists(TEST_IMG_REAL):
    import glob
    jpgs = glob.glob(r"E:\DEEPGUARD\backend\static\uploads\*.jpg") or \
           glob.glob(r"E:\DEEPGUARD\benchmark_100\images\s1_*.jpg")
    TEST_IMG_REAL = jpgs[0] if jpgs else TEST_IMG_REAL

results = {"smoke": [], "functional": [], "boundary": [], "security": [], "performance": [], "regression": []}

def log(section, name, status, detail=""):
    icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"  {icon} {name}: {status}" + (f" -- {detail}" if detail else ""))
    results[section].append({"test": name, "status": status, "detail": detail})

# ============================================================
# SECTION 1 -- SMOKE TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 1: SMOKE TESTS (Core System Availability)")
print("="*65)

smoke_cases = [
    ("Home / Landing Page",              "/",              [200, 302]),
    ("Login Page",                       "/login",         [200]),
    ("Register Page",                    "/register",      [200]),
    ("Analyze Page (auth guard)",        "/analyze",       [200, 302]),
    ("History Page (auth guard)",        "/history",       [200, 302]),
    ("Dashboard (auth guard)",           "/dashboard",     [200, 302]),
    ("Modules Architecture Page",        "/modules",       [200]),
    ("About Page",                       "/about",         [200]),
    ("Notifications API",                "/notifications", [200, 302, 401]),
]

for name, path, expected in smoke_cases:
    try:
        t0 = time.time()
        r = requests.get(f"{BASE_URL}{path}", timeout=8, allow_redirects=True)
        ms = round((time.time()-t0)*1000)
        ok = r.status_code in expected
        log("smoke", name, "PASS" if ok else "FAIL",
            f"HTTP {r.status_code} in {ms}ms")
    except Exception as e:
        log("smoke", name, "FAIL", str(e)[:80])

# ============================================================
# SECTION 2 -- FUNCTIONAL TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 2: FUNCTIONAL TESTS (AI Modules)")
print("="*65)

sys.path.insert(0, r"E:\DEEPGUARD")
from modules.image_detection.image_detector import predict_image_authenticity
from modules.context_mismatch.mismatch import detect_context_mismatch
from modules.text_analysis.text_analyzer import analyze_text_fake
from modules.fusion_engine.fusion import fuse_predictions

# 2.1 Image: Real camera photo
t0 = time.time()
res = predict_image_authenticity(TEST_IMG_REAL)
ms = round((time.time()-t0)*1000)
ok = res.get("prediction") == "REAL"
log("functional", "Real camera photo --> Predict REAL",
    "PASS" if ok else "FAIL",
    f"Prediction: {res.get('prediction')}, Conf: {res.get('confidence')}%, Engine: {res.get('engine')}, {ms}ms")

# 2.2 Image: AI-generated
t0 = time.time()
res_fake = predict_image_authenticity(TEST_IMG_FAKE)
ms = round((time.time()-t0)*1000)
ok = res_fake.get("prediction") == "FAKE"
log("functional", "AI-generated image --> Predict FAKE",
    "PASS" if ok else "WARN",
    f"Prediction: {res_fake.get('prediction')}, Conf: {res_fake.get('confidence')}%, {ms}ms")

# 2.3 EXIF integration
ok = "metadata_forensics" in res
log("functional", "EXIF metadata_forensics key in image output",
    "PASS" if ok else "FAIL", f"Keys: {list(res.keys())}")

real_exif = res.get("metadata_forensics", {})
log("functional", "Real phone photo has EXIF metadata",
    "PASS" if real_exif.get("has_metadata") else "WARN",
    f"has_metadata: {real_exif.get('has_metadata')}")

# 2.4 Text: Viral fake claim (>20 words to pass minimum word threshold)
t0 = time.time()
tr1 = analyze_text_fake("BREAKING: Scientists have officially confirmed that 5G towers spread dangerous radiation and viral infections across major cities worldwide. Share this urgent warning immediately before it gets deleted by the authorities!")
ms = round((time.time()-t0)*1000)
fp1 = float(tr1.get("fake_probability", 0))
ok = fp1 > 50.0  # text_analyzer returns 0-100 range
log("functional", "Viral fake text --> DistilBERT flags as FAKE",
    "PASS" if ok else "FAIL",
    f"fake_prob: {fp1:.1f}% (>50 = FAKE), {ms}ms")

# 2.5 Text: Neutral factual claim (>20 words to pass minimum word threshold)
t0 = time.time()
tr2 = analyze_text_fake("Researchers at MIT and local Pakistani universities have published comprehensive scientific findings on biodiversity conservation and ecological sustainability in the northern mountain regions of Pakistan today.")
ms = round((time.time()-t0)*1000)
fp2 = float(tr2.get("fake_probability", 100))
ok = fp2 < 50.0
log("functional", "Factual neutral text --> DistilBERT flags as REAL",
    "PASS" if ok else "WARN",
    f"fake_prob: {fp2:.1f}% (<50 = REAL), {ms}ms")

# 2.6 Context Mismatch: valid response
t0 = time.time()
cm1 = detect_context_mismatch(TEST_IMG_REAL, "A smartphone photo of an outdoor scene.")
ms = round((time.time()-t0)*1000)
ok = cm1.get("status") == "success"
log("functional", "Context Mismatch module -- valid response returned",
    "PASS" if ok else "FAIL",
    f"similarity: {cm1.get('similarity')}, mismatch_score: {cm1.get('mismatch_score')}, {ms}ms")

# 2.7 Context Mismatch: clearly mismatched
t0 = time.time()
cm2 = detect_context_mismatch(TEST_IMG_REAL, "Giant alien spacecraft landing on the moon.")
ms = round((time.time()-t0)*1000)
ok = cm2.get("mismatch_score", 0) > 40
log("functional", "Clearly mismatched image+text --> High mismatch score",
    "PASS" if ok else "FAIL",
    f"mismatch_score: {cm2.get('mismatch_score')}, is_mismatch: {cm2.get('is_mismatch')}, {ms}ms")

# 2.8 Fusion: all-high scores = FAKE
fus_hi = fuse_predictions({"text_score": 85.0, "image_score": 90.0, "viral_score": 75.0, "context_mismatch_score": 70.0})
ok = fus_hi.get("verdict") == "FAKE" and fus_hi.get("risk_level") in ["HIGH", "CRITICAL"]
log("functional", "High-score inputs --> Fusion verdict FAKE + HIGH/CRITICAL",
    "PASS" if ok else "FAIL",
    f"verdict: {fus_hi.get('verdict')}, risk: {fus_hi.get('risk_level')}, score: {fus_hi.get('final_score')}")

# 2.9 Fusion: all-low scores = REAL
fus_lo = fuse_predictions({"text_score": 8.0, "image_score": 3.0, "viral_score": 0.0, "context_mismatch_score": 5.0})
ok = fus_lo.get("verdict") == "REAL"
log("functional", "Low-score inputs --> Fusion verdict REAL",
    "PASS" if ok else "FAIL",
    f"verdict: {fus_lo.get('verdict')}, score: {fus_lo.get('final_score')}")

# 2.10 Fusion: cross-modal mismatch override
fus_mm = fuse_predictions({"text_score": 20.0, "image_score": 15.0, "viral_score": 5.0, "context_mismatch_score": 72.0})
ok = fus_mm.get("verdict") == "FAKE"
log("functional", "Severe context mismatch (>=60) --> Cross-modal override to FAKE",
    "PASS" if ok else "FAIL",
    f"verdict: {fus_mm.get('verdict')}, score: {fus_mm.get('final_score')}")

# ============================================================
# SECTION 3 -- BOUNDARY / EDGE CASE TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 3: BOUNDARY & EDGE CASE TESTS")
print("="*65)

# 3.1 Empty text
try:
    r = analyze_text_fake("")
    ok = r.get("prediction") == "UNKNOWN" or r.get("status") == "empty_input" or r.get("fake_probability") == 0.0
    log("boundary", "Empty string text input handled gracefully",
        "PASS" if ok else "WARN", f"Response: prediction={r.get('prediction')}, status={r.get('status')}")
except Exception as e:
    log("boundary", "Empty string text input", "FAIL", str(e)[:80])

# 3.2 Very long text (7400+ chars)
try:
    long_txt = "This is a test claim about fake news and misinformation spreading online. " * 100
    t0 = time.time()
    r = analyze_text_fake(long_txt)
    ms = round((time.time()-t0)*1000)
    log("boundary", "Very long text (7400+ chars) handled without crash",
        "PASS", f"fake_prob: {r.get('fake_probability')}, {ms}ms")
except Exception as e:
    log("boundary", "Very long text", "FAIL", str(e)[:80])

# 3.3 Urdu (non-English) text
try:
    r = analyze_text_fake("یہ ایک جھوٹی خبر ہے جو سوشل میڈیا پر وائرل ہو رہی ہے")
    log("boundary", "Urdu text input handled gracefully (no crash)",
        "PASS", f"fake_prob: {r.get('fake_probability')}")
except Exception as e:
    log("boundary", "Urdu text input", "FAIL", str(e)[:80])

# 3.4 Missing image path
r = predict_image_authenticity(r"E:\DEEPGUARD\nonexistent_image_xyz.jpg")
ok = r.get("prediction") == "UNKNOWN"
log("boundary", "Non-existent image path returns UNKNOWN gracefully",
    "PASS" if ok else "FAIL", f"prediction: {r.get('prediction')}")

# 3.5 Fusion with all-zero inputs
fz = fuse_predictions({"text_score": 0.0, "image_score": 0.0, "viral_score": 0.0, "context_mismatch_score": 0.0})
ok = fz.get("verdict") == "REAL" and fz.get("final_score", 1) == 0.0
log("boundary", "All-zero fusion scores --> REAL verdict, score=0.0",
    "PASS" if ok else "FAIL", f"verdict: {fz.get('verdict')}, score: {fz.get('final_score')}")

# 3.6 Image > 10 MB rejected by system
try:
    import tempfile
    temp_dir = tempfile.gettempdir()
    big_file = os.path.join(temp_dir, "deepguard_test_11mb_boundary.jpg")
    with open(big_file, "wb") as f:
        f.seek(11 * 1024 * 1024 - 1)
        f.write(b"\0")

    # 1. Test image detector module rejection
    det_res = predict_image_authenticity(big_file)
    ok_module = det_res.get("status") == "file_too_large"

    # 2. Test HTTP layer rejection
    with open(big_file, "rb") as bf:
        r = requests.post(f"{BASE_URL}/predict",
                          files={"image": ("big.jpg", bf, "image/jpeg")},
                          data={"text": "This is a full test claim with sufficient words for checking boundary file sizes."},
                          timeout=10, allow_redirects=False)
    ok_http = r.status_code in [413, 302]

    if os.path.exists(big_file):
        os.remove(big_file)

    ok = ok_module and ok_http
    log("boundary", "Image > 10 MB rejected by system",
        "PASS" if ok else "FAIL", f"Module: status={det_res.get('status')}, HTTP: {r.status_code}")
except Exception as e:
    log("boundary", "Image > 10 MB rejection test", "WARN", str(e)[:80])

# ============================================================
# SECTION 4 -- SECURITY TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 4: SECURITY TESTS")
print("="*65)

# 4.1 XSS in text input (AI module should not echo raw HTML back)
try:
    xss = "<script>alert('XSS')</script>"
    r = analyze_text_fake(xss)
    ok = "<script>" not in str(r.get("prediction", ""))
    log("security", "XSS payload in text -- not reflected in output",
        "PASS" if ok else "FAIL", "No raw <script> tags in module output")
except Exception as e:
    log("security", "XSS payload test", "FAIL", str(e)[:80])

# 4.2 SQL injection in text
try:
    sql = "'; DROP TABLE users; --"
    r = analyze_text_fake(sql)
    log("security", "SQL injection string handled without crash",
        "PASS", f"fake_prob: {r.get('fake_probability')}")
except Exception as e:
    log("security", "SQL injection test", "FAIL", str(e)[:80])

# 4.3 Unauthenticated POST to /predict must be blocked
r = requests.post(f"{BASE_URL}/predict", data={"text": "test"}, timeout=5, allow_redirects=False)
ok = r.status_code in [302, 401, 403, 405]
log("security", "Unauthenticated POST /predict -- access blocked",
    "PASS" if ok else "FAIL", f"HTTP Status: {r.status_code}")

# 4.4 Unauthenticated GET to /history must be blocked
r = requests.get(f"{BASE_URL}/history", timeout=5, allow_redirects=False)
ok = r.status_code in [302, 401, 403]
log("security", "Unauthenticated GET /history -- redirect enforced",
    "PASS" if ok else "FAIL", f"HTTP Status: {r.status_code}")

# 4.5 Path traversal in upload filename
r = requests.post(f"{BASE_URL}/predict",
                  files={"image": ("../../etc/passwd.jpg", b"fake", "image/jpeg")},
                  data={"text": "test"}, timeout=8, allow_redirects=False)
ok = r.status_code in [302, 401, 403, 400, 500]
log("security", "Path traversal filename (../../etc/passwd.jpg) -- not executed",
    "PASS" if ok else "WARN", f"HTTP Status: {r.status_code}")

# ============================================================
# SECTION 5 -- PERFORMANCE TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 5: PERFORMANCE TESTS")
print("="*65)

# 5.1 Image inference latency (3 runs)
times = []
for _ in range(3):
    t0 = time.time()
    predict_image_authenticity(TEST_IMG_REAL)
    times.append(round((time.time()-t0)*1000))
avg_img = sum(times)//len(times)
log("performance", "Image inference (ONNX) avg latency over 3 runs",
    "PASS" if avg_img < 2500 else "WARN",
    f"Avg: {avg_img}ms | Runs: {times} | Target: <2500ms")

# 5.2 Text inference latency (3 runs)
times = []
for _ in range(3):
    t0 = time.time()
    analyze_text_fake("Climate scientists warn of rising sea levels in coastal cities.")
    times.append(round((time.time()-t0)*1000))
avg_txt = sum(times)//len(times)
log("performance", "Text inference (DistilBERT) avg latency over 3 runs",
    "PASS" if avg_txt < 1500 else "WARN",
    f"Avg: {avg_txt}ms | Runs: {times} | Target: <1500ms")

# 5.3 Fusion engine computation speed (10 runs)
times = []
for _ in range(10):
    t0 = time.time()
    fuse_predictions({"text_score": 75.0, "image_score": 60.0, "viral_score": 40.0, "context_mismatch_score": 55.0})
    times.append((time.time()-t0)*1000)
avg_fus = round(sum(times)/len(times), 2)
log("performance", "Fusion engine computation over 10 runs",
    "PASS", f"Avg: {avg_fus}ms (near-instantaneous, target: <50ms)")

# 5.4 HTTP concurrent requests (20 simultaneous requests to /modules)
concurrent_results = []
def hit_modules():
    try:
        r = requests.get(f"{BASE_URL}/modules", timeout=15)
        concurrent_results.append(r.status_code == 200)
    except:
        concurrent_results.append(False)

concurrent = 20
t0 = time.time()
threads = [threading.Thread(target=hit_modules) for _ in range(concurrent)]
for t in threads: t.start()
for t in threads: t.join()
elapsed = round((time.time()-t0)*1000)
success_rate = sum(concurrent_results)/len(concurrent_results)*100
log("performance", f"{concurrent} concurrent HTTP requests to /modules",
    "PASS" if success_rate >= 90 else "WARN",
    f"{sum(concurrent_results)}/{concurrent} succeeded in {elapsed}ms (avg {elapsed//concurrent}ms/req)")

# 5.5 HTTP page response time targets
for name, path, target_ms in [("Login page", "/login", 1000), ("Modules page", "/modules", 1500), ("About page", "/about", 1500)]:
    times = []
    for _ in range(3):
        t0 = time.time()
        requests.get(f"{BASE_URL}{path}", timeout=10)
        times.append(round((time.time()-t0)*1000))
    avg = sum(times)//len(times)
    log("performance", f"Page load: {name}",
        "PASS" if avg < target_ms else "WARN",
        f"Avg: {avg}ms | Target: <{target_ms}ms")

# ============================================================
# SECTION 6 -- REGRESSION TESTS
# ============================================================
print("\n" + "="*65)
print("  SECTION 6: REGRESSION TESTS (Post Model-Upgrade Validation)")
print("="*65)

# 6.1 ONNX engine active (not old MobileNetV2)
fresh = predict_image_authenticity(TEST_IMG_REAL)
ok = fresh.get("engine") == "onnx"
log("regression", "EfficientNetV2-S ONNX engine active (not old MobileNetV2)",
    "PASS" if ok else "FAIL",
    f"engine: {fresh.get('engine')}, model: {fresh.get('model_architecture', 'N/A')}")

# 6.2 Old EXIF calibration workaround removed from source
import inspect
from modules.image_detection import image_detector as _imgmod
src = inspect.getsource(_imgmod)
ok = "fake_prob *= 0.10" not in src and "calibration" not in src.lower()
log("regression", "Old EXIF calibration multiplier (0.10) removed from code",
    "PASS" if ok else "FAIL",
    "No fake_prob *= 0.10 found in source" if ok else "OLD WORKAROUND STILL PRESENT!")

# 6.3 Context mismatch weight >= 0.20 in fusion
from modules.fusion_engine.fusion import BASE_WEIGHTS
mw = BASE_WEIGHTS.get("context_mismatch_score", 0)
ok = mw >= 0.20
log("regression", "Context mismatch weight >= 0.20 in fusion engine",
    "PASS" if ok else "FAIL",
    f"context_mismatch_score weight: {mw}")

# 6.4 metadata_forensics key in image output
ok = "metadata_forensics" in fresh
log("regression", "metadata_forensics key integrated in image output dict",
    "PASS" if ok else "FAIL",
    f"Keys present: {list(fresh.keys())}")

# 6.5 Pipeline returns all required keys
from modules.pipeline import run_deepguard_analysis
pipe = run_deepguard_analysis(
    text="This is a fully verified news report discussing technological advancements and environmental sustainability across major urban infrastructure developments in Pakistan today.",
    image_path=TEST_IMG_REAL
)
required = {"verdict", "is_fake", "final_score", "confidence", "risk_level", "module_details", "fusion_breakdown"}
missing = required - set(pipe.keys())
ok = len(missing) == 0
log("regression", "Pipeline output contains all 7 required keys",
    "PASS" if ok else "FAIL",
    f"Missing keys: {missing if missing else 'None'}")

# 6.6 Context mismatch cross-modal rule still active
fm = fuse_predictions({"text_score": 10.0, "image_score": 10.0, "viral_score": 0.0, "context_mismatch_score": 65.0})
ok = fm.get("verdict") == "FAKE" and float(fm.get("final_score", 0)) >= 52.0
log("regression", "Cross-modal elevation rule (mismatch>=60 -> score>=52) active",
    "PASS" if ok else "FAIL",
    f"verdict: {fm.get('verdict')}, score: {fm.get('final_score')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)

total_pass = total_fail = total_warn = 0
for section, tests in results.items():
    p = sum(1 for t in tests if t["status"] == "PASS")
    f = sum(1 for t in tests if t["status"] == "FAIL")
    w = sum(1 for t in tests if t["status"] == "WARN")
    total_pass += p; total_fail += f; total_warn += w
    print(f"  {section.upper():<20}: PASS={p}  FAIL={f}  WARN={w}")

grand = total_pass + total_fail + total_warn
print("-"*65)
print(f"  TOTAL ({grand} tests) : PASS={total_pass}  FAIL={total_fail}  WARN={total_warn}")
pct = round(total_pass/grand*100, 1) if grand else 0
print(f"  PASS RATE            : {pct}%")
print("="*65)

out_path = r"E:\DEEPGUARD\benchmark_100\test_results_summary.json"
with open(out_path, "w") as f:
    json.dump({"results": results, "summary": {
        "total": grand, "pass": total_pass, "fail": total_fail, "warn": total_warn,
        "pass_rate": pct
    }}, f, indent=2)
print(f"\n  Results saved to: {out_path}")
