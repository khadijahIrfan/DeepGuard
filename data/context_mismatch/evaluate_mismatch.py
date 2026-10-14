import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = r"E:\DeepGuard"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.context_mismatch.mismatch import clip_similarity


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = r"E:\DeepGuard\data\context_mismatch"

LABELS_PATH = os.path.join(
    BASE_DIR,
    "labels.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images"
)

RESULTS_CSV = os.path.join(
    BASE_DIR,
    "threshold_results.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "confusion_matrix.png"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(LABELS_PATH)

print("=" * 60)
print("CONTEXT MISMATCH EVALUATION")
print("=" * 60)

print("Total samples:", len(df))
print("Matched:", (df["label"] == 0).sum())
print("Mismatched:", (df["label"] == 1).sum())


# --------------------------------------------------
# Calculate CLIP similarity
# --------------------------------------------------

similarities = []
valid_labels = []
valid_ids = []

print("\nCalculating CLIP similarities...")
print("This may take some time on CPU.\n")

for index, row in df.iterrows():

    image_id = str(row["image_id"])
    caption = str(row["caption"])
    label = int(row["label"])

    image_path = os.path.join(
        IMAGE_DIR,
        f"{image_id}.jpg"
    )

    if not os.path.exists(image_path):
        print(
            f"Skipping missing image: {image_id}"
        )
        continue

    try:

        similarity = clip_similarity(
            image_path=image_path,
            text=caption
        )

        similarities.append(similarity)
        valid_labels.append(label)
        valid_ids.append(image_id)

        if (len(similarities) % 25) == 0:
            print(
                f"Processed: "
                f"{len(similarities)}/{len(df)}"
            )

    except Exception as e:

        print(
            f"Error processing {image_id}: {e}"
        )


# --------------------------------------------------
# Convert to arrays
# --------------------------------------------------

similarities = np.array(similarities)
labels = np.array(valid_labels)


print("\n" + "=" * 60)
print("SIMILARITY CALCULATION COMPLETE")
print("=" * 60)

print("Valid samples:", len(similarities))


# --------------------------------------------------
# Threshold evaluation
# --------------------------------------------------

thresholds = np.round(
    np.arange(0.15, 0.301, 0.01),
    2
)

results = []

for threshold in thresholds:

    # Lower similarity = mismatch
    predictions = (
        similarities < threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# --------------------------------------------------
# Best threshold
# --------------------------------------------------

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

best_accuracy = float(
    best_row["accuracy"]
)

best_precision = float(
    best_row["precision"]
)

best_recall = float(
    best_row["recall"]
)

best_f1 = float(
    best_row["f1"]
)


# --------------------------------------------------
# Final predictions
# --------------------------------------------------

final_predictions = (
    similarities < best_threshold
).astype(int)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

report = classification_report(
    labels,
    final_predictions,
    target_names=[
        "MATCH",
        "MISMATCH"
    ],
    zero_division=0
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    labels,
    final_predictions
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_accuracy:.4f}"
)

print(
    f"Precision : {best_precision:.4f}"
)

print(
    f"Recall    : {best_recall:.4f}"
)

print(
    f"F1 Score  : {best_f1:.4f}"
)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)


# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    f"Context Mismatch Confusion Matrix\n"
    f"Threshold = {best_threshold:.2f}"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["MATCH", "MISMATCH"]
)

plt.yticks(
    [0, 1],
    ["MATCH", "MISMATCH"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    "Threshold results:",
    RESULTS_CSV
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)

print("=" * 60)