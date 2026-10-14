import sys
import pathlib
import pandas as pd
from tqdm import tqdm

# Add root directory to path
BASE_DIR = pathlib.Path(r"E:\DEEPGUARD")
sys.path.insert(0, str(BASE_DIR))

from modules.pipeline import run_deepguard_analysis

BENCH_DIR = BASE_DIR / "benchmark_100"
INPUT_CSV = BENCH_DIR / "deepguard_100_multimodal_benchmark.csv"
OUTPUT_CSV = BENCH_DIR / "deepguard_100_evaluated_results.csv"

def run_evaluation():
    print("=" * 70)
    print("   DEEPGUARD 100-SAMPLE MULTIMODAL BENCHMARK EVALUATOR")
    print("=" * 70)
    
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples across 4 scenarios.\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing pairs"):
        text = str(row['text'])
        img_path = str(row['image_file'])
        
        try:
            res = run_deepguard_analysis(text=text, image_path=img_path)
            
            # Module scores
            img_details = res.get('module_details', {}).get('image_detection', {})
            text_details = res.get('module_details', {}).get('text_analysis', {})
            mismatch_details = res.get('module_details', {}).get('context_mismatch', {})
            
            img_score = img_details.get('manipulation_score', 0.0)
            text_score = text_details.get('fake_probability', 50.0)
            mismatch_score = mismatch_details.get('mismatch_score', 0.0)
            final_verdict = res.get('verdict', 'UNKNOWN')
            
            # Check correctness
            expected = str(row['expected_verdict']).strip().upper()
            predicted = str(final_verdict).strip().upper()
            is_correct = "YES" if (predicted == expected or (expected == "FAKE" and predicted in ["FAKE", "SUSPICIOUS"])) else "NO"
            
            df.at[idx, 'image_score'] = round(img_score, 2)
            df.at[idx, 'text_score'] = round(text_score, 2)
            df.at[idx, 'mismatch_score'] = round(mismatch_score, 2)
            df.at[idx, 'final_verdict'] = predicted
            df.at[idx, 'correct'] = is_correct
            
        except Exception as e:
            print(f"Error on sample {row['id']}: {e}")
            df.at[idx, 'final_verdict'] = "ERROR"
            df.at[idx, 'correct'] = "NO"

    # Save completed evaluation CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved evaluated results to: {OUTPUT_CSV}\n")
    
    # ------------------ PERFORMANCE SUMMARY ------------------
    print("=" * 70)
    print("                PER-SCENARIO ACCURACY BREAKDOWN")
    print("=" * 70)
    
    scenarios = [
        (1, "Scenario 1: Real Image + Real Text (Matching)"),
        (2, "Scenario 2: Real Image + Fake Text"),
        (3, "Scenario 3: Fake (AI) Image + Text"),
        (4, "Scenario 4: Real Image + Real Text (Context Mismatch)")
    ]
    
    total_correct = (df['correct'] == 'YES').sum()
    
    for s_id, s_name in scenarios:
        sub = df[df['scenario'] == s_id]
        corr = (sub['correct'] == 'YES').sum()
        total = len(sub)
        acc = (corr / total) * 100 if total > 0 else 0
        print(f"  {s_name:<54} : {corr:2d}/{total:2d} ({acc:5.1f}%)")
        
    print("-" * 70)
    print(f"  OVERALL SYSTEM ACCURACY (100 Samples)                  : {total_correct:2d}/100 ({total_correct:5.1f}%)\n")
    
    # ------------------ CONFUSION MATRIX ------------------
    print("=" * 70)
    print("                       CONFUSION MATRIX")
    print("=" * 70)
    
    # Map predictions to REAL vs FAKE
    y_true = df['expected_verdict'].apply(lambda x: 'REAL' if x == 'REAL' else 'FAKE')
    y_pred = df['final_verdict'].apply(lambda x: 'REAL' if x == 'REAL' else 'FAKE')
    
    tp = ((y_true == 'FAKE') & (y_pred == 'FAKE')).sum()
    tn = ((y_true == 'REAL') & (y_pred == 'REAL')).sum()
    fp = ((y_true == 'REAL') & (y_pred == 'FAKE')).sum()
    fn = ((y_true == 'FAKE') & (y_pred == 'REAL')).sum()
    
    print(f"                Predicted REAL    Predicted FAKE")
    print(f"  Actual REAL:        {tn:2d} (TN)            {fp:2d} (FP)")
    print(f"  Actual FAKE:        {fn:2d} (FN)            {tp:2d} (TP)\n")
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Precision : {precision:.2f}%")
    print(f"  Recall    : {recall:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%\n")
    
    # ------------------ FAILURE ANALYSIS ------------------
    print("=" * 70)
    print("                  FAILURE CASE EXAMPLES (For Thesis Defense)")
    print("=" * 70)
    failures = df[df['correct'] == 'NO']
    if len(failures) == 0:
        print("  No failures detected!")
    else:
        for idx, (_, f_row) in enumerate(failures.head(5).iterrows(), 1):
            print(f"  Failure #{idx} (Scenario {f_row['scenario']}) - Sample ID: {f_row['id']}")
            print(f"    Text: \"{f_row['text'][:70]}...\"")
            print(f"    Expected: {f_row['expected_verdict']} | Predicted: {f_row['final_verdict']}")
            print(f"    Scores -> Img: {f_row['image_score']} | Txt: {f_row['text_score']} | Mismatch: {f_row['mismatch_score']}")
            print()

if __name__ == "__main__":
    run_evaluation()
