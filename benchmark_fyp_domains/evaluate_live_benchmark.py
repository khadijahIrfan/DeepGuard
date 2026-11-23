"""
Empirical Benchmark Evaluation for DeepGuard
=============================================
Runs live inference using DeepGuardPipeline on all 100 benchmark test items:
  - Scenario 1: Real Image + Real Text (25 samples)
  - Scenario 2: Real Image + Fake Text (25 samples)
  - Scenario 3: AI-Generated Image + Text (25 samples)
  - Scenario 4: Real Image + Real Text Context Mismatch (25 samples)

Strict Constraints:
- Stored and run ONLY inside E:\DEEPGUARD\benchmark_fyp_domains\
- Does NOT alter any application files or models.
"""

import os
import sys
import time
import json
import pathlib
import pandas as pd

# Set paths
BENCHMARK_DIR = pathlib.Path(r"E:\DEEPGUARD\benchmark_fyp_domains")
ROOT_DIR = pathlib.Path(r"E:\DEEPGUARD")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.pipeline import DeepGuardPipeline

def run_evaluation():
    csv_path = BENCHMARK_DIR / "deepguard_fyp_benchmark_results.csv"
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Loaded {total_samples} benchmark samples from {csv_path.name}")
    print("Initializing DeepGuard Pipeline...")
    pipeline = DeepGuardPipeline()
    print("Pipeline initialized successfully. Starting live inference...\n")

    results = []
    scenario_counts = {1: {"correct": 0, "total": 0},
                       2: {"correct": 0, "total": 0},
                       3: {"correct": 0, "total": 0},
                       4: {"correct": 0, "total": 0}}

    start_time_all = time.time()

    for idx, row in df.iterrows():
        sample_id = int(row["Sample_ID"])
        scenario_id = int(row["Scenario_ID"])
        text = str(row["Text_Content"])
        img_path = str(row["Image_Local_Path"])
        
        # Ground truth
        gt_verdict = str(row["Expected_Final_Verdict"]).strip().upper()
        is_ground_truth_fake = not ("REAL" in gt_verdict and "MISINFORMATION" not in gt_verdict and "MISMATCH" not in gt_verdict)
        if scenario_id == 1:
            is_ground_truth_fake = False
        else:
            is_ground_truth_fake = True

        t0 = time.time()
        try:
            pred = pipeline.analyze(text=text, image_path=img_path)
            predicted_is_fake = pred.get("is_fake", False)
            predicted_verdict = pred.get("verdict", "UNKNOWN")
            confidence = pred.get("confidence", 0.0)
            
            fb = pred.get("fusion_breakdown", {})
            text_score = fb.get("text_score", {}).get("raw_score", 0.0)
            image_score = fb.get("image_score", {}).get("raw_score", 0.0)
            mismatch_score = fb.get("context_mismatch_score", {}).get("raw_score", 0.0)
            status = "SUCCESS"
        except Exception as e:
            predicted_is_fake = False
            predicted_verdict = f"ERROR: {str(e)[:30]}"
            confidence = 0.0
            text_score = 0.0
            image_score = 0.0
            mismatch_score = 0.0
            status = "ERROR"

        dur = round((time.time() - t0) * 1000, 1)

        is_correct = (predicted_is_fake == is_ground_truth_fake)

        scenario_counts[scenario_id]["total"] += 1
        if is_correct:
            scenario_counts[scenario_id]["correct"] += 1

        results.append({
            "Sample_ID": sample_id,
            "Scenario_ID": scenario_id,
            "Scenario_Name": row["Scenario_Name"],
            "Domain": row["Domain"],
            "Ground_Truth_Is_Fake": is_ground_truth_fake,
            "Predicted_Is_Fake": predicted_is_fake,
            "Predicted_Verdict": predicted_verdict,
            "Confidence": confidence,
            "Text_Score": text_score,
            "Image_Score": image_score,
            "Mismatch_Score": mismatch_score,
            "Is_Correct": is_correct,
            "Latency_ms": dur,
            "Status": status
        })

        if sample_id % 10 == 0 or sample_id == total_samples:
            print(f"Processed [{sample_id:03d}/{total_samples:03d}] (Scenario {scenario_id}) | Accuracy so far: {sum(s['correct'] for s in scenario_counts.values())}/{sum(s['total'] for s in scenario_counts.values())}")

    total_time = round(time.time() - start_time_all, 2)
    print(f"\nFinished live evaluation of all {total_samples} samples in {total_time}s\n")

    # Save detailed empirical results
    res_df = pd.DataFrame(results)
    out_csv = BENCHMARK_DIR / "empirical_live_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[+] Saved detailed per-sample empirical results to: {out_csv}")

    # Summary table
    s1_corr, s1_tot = scenario_counts[1]["correct"], scenario_counts[1]["total"]
    s2_corr, s2_tot = scenario_counts[2]["correct"], scenario_counts[2]["total"]
    s3_corr, s3_tot = scenario_counts[3]["correct"], scenario_counts[3]["total"]
    s4_corr, s4_tot = scenario_counts[4]["correct"], scenario_counts[4]["total"]
    tot_corr = s1_corr + s2_corr + s3_corr + s4_corr
    tot_all = s1_tot + s2_tot + s3_tot + s4_tot

    summary = {
        "S1": {"correct": s1_corr, "total": s1_tot, "accuracy": round(s1_corr / s1_tot * 100, 1)},
        "S2": {"correct": s2_corr, "total": s2_tot, "accuracy": round(s2_corr / s2_tot * 100, 1)},
        "S3": {"correct": s3_corr, "total": s3_tot, "accuracy": round(s3_corr / s3_tot * 100, 1)},
        "S4": {"correct": s4_corr, "total": s4_tot, "accuracy": round(s4_corr / s4_tot * 100, 1)},
        "Overall": {"correct": tot_corr, "total": tot_all, "accuracy": round(tot_corr / tot_all * 100, 1)}
    }

    out_json = BENCHMARK_DIR / "empirical_live_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("Table 20: Multimodal Evaluation Results by Test Scenario")
    print("="*60)
    print(f"{'Scenario':<30}\t{'Correct':<8}\t{'Total':<8}\t{'Accuracy'}")
    print(f"{'S1: real image, real text':<30}\t{s1_corr:<8}\t{s1_tot:<8}\t{summary['S1']['accuracy']}%")
    print(f"{'S2: real image, fake text':<30}\t{s2_corr:<8}\t{s2_tot:<8}\t{summary['S2']['accuracy']}%")
    print(f"{'S3: AI-generated image':<30}\t{s3_corr:<8}\t{s3_tot:<8}\t{summary['S3']['accuracy']}%")
    print(f"{'S4: context mismatch':<30}\t{s4_corr:<8}\t{s4_tot:<8}\t{summary['S4']['accuracy']}%")
    print("-" * 60)
    print(f"{'Overall':<30}\t{tot_corr:<8}\t{tot_all:<8}\t{summary['Overall']['accuracy']}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()
