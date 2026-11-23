import json
import sys
from pathlib import Path

BASE_DIR = Path(r"E:\DEEPGUARD")
sys.path.insert(0, str(BASE_DIR))

from modules.pipeline import run_deepguard_analysis

samples = [
    {
        "id": "Sample 1",
        "title": "TrendX Jawad Ahmad / PTI Political Card",
        "path": r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\.user_uploaded\media_1794219951412_183b2c37.jpg"
    },
    {
        "id": "Sample 2",
        "title": "Startup Pakistan BYD Cars News Card",
        "path": r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\.user_uploaded\media_1794219951417_85df2e97.jpg"
    },
    {
        "id": "Sample 3",
        "title": "Lahore Flood Emergency Hoax Card (Dry Mosque Mismatch)",
        "path": r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\.user_uploaded\media_1794219951464_07a31272.jpg"
    },
    {
        "id": "Sample 4",
        "title": "End of the World / American Scientist Hoax with AI Art",
        "path": r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\.user_uploaded\media_1794219951603_1e640e2b.jpg"
    },
    {
        "id": "Sample 5",
        "title": "Live Breaking News Gaza / Israel Banner",
        "path": r"C:\Users\GA LAPTOP\.gemini\antigravity\brain\a405a7c6-d77f-4538-a4e6-e08e46e90da8\.user_uploaded\media_1794219951632_d86d2e74.jpg"
    }
]

print("=" * 80)
print("DEEPGUARD COMPREHENSIVE VERIFICATION BENCHMARK: 5 SAMPLE TEST IMAGES")
print("=" * 80)

for s in samples:
    print(f"\nEvaluating: {s['id']} - {s['title']}")
    print(f"Path: {s['path']}")
    
    res = run_deepguard_analysis(text=None, image_path=s["path"], is_grounded=True)
    
    print("-" * 50)
    print(f"Extracted Grounded Text: {res.get('analyzed_text', '')[:120]}...")
    print(f"Verdict: {res['verdict']}")
    print(f"Final Score (Deception Prob): {res['final_score']}%")
    print(f"Confidence: {res['confidence']}%")
    print(f"Risk Level: {res['risk_level']}")
    print(f"Modules Executed: {res['modules_executed']}")
    print("Module Scores:")
    for mod, details in res.get("fusion_breakdown", {}).items():
        print(f"   • {mod}: raw = {details['raw_score']}, weight = {details['weight_percentage']}%")
    print("-" * 50)

print("\nBenchmark completed!")
