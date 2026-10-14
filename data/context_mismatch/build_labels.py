import os
import pandas as pd

BASE_DIR = r"E:\DeepGuard\data\context_mismatch"

pairs_csv = os.path.join(BASE_DIR, "selected_pairs.csv")
images_dir = os.path.join(BASE_DIR, "images")

df = pd.read_csv(pairs_csv)

available_images = {
    os.path.splitext(f)[0]
    for f in os.listdir(images_dir)
}

df = (
    df[df["id"].astype(str).isin(available_images)]
    .copy()
    .reset_index(drop=True)
)

df.reset_index(drop=True, inplace=True)

# ------------------------
# MATCHED PAIRS
# ------------------------
matched = pd.DataFrame({
    "image_id": df["id"],
    "caption": df["clean_title"],
    "label": 0
})

# ------------------------
# MISMATCHED PAIRS
# ------------------------
shuffled = df["clean_title"].sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

mismatched = pd.DataFrame({
    "image_id": df["id"].values,
    "caption": shuffled,
    "label": 1
})

# ensure image not paired with own caption
for i in range(len(mismatched)):
    if mismatched.loc[i, "caption"] == matched.loc[i, "caption"]:
        swap_idx = (i + 1) % len(mismatched)

        temp = mismatched.loc[i, "caption"]
        mismatched.loc[i, "caption"] = mismatched.loc[swap_idx, "caption"]
        mismatched.loc[swap_idx, "caption"] = temp

labels = pd.concat(
    [matched, mismatched],
    ignore_index=True
)

labels.to_csv(
    os.path.join(BASE_DIR, "labels.csv"),
    index=False
)

print("=" * 60)
print("labels.csv created")
print("Matched pairs:", len(matched))
print("Mismatched pairs:", len(mismatched))
print("Total pairs:", len(labels))
print("=" * 60)