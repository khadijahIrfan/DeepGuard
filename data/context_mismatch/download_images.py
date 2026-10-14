import os
import time
import pandas as pd
import requests

BASE_DIR = r"E:\DeepGuard\data\context_mismatch"
CSV_PATH = os.path.join(BASE_DIR, "selected_pairs.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

total = len(df)
downloaded = 0
skipped = 0
failed = 0

print("=" * 60)
print(f"Total selected images: {total}")
print(f"Image directory: {IMAGE_DIR}")
print("=" * 60)

for index, row in df.iterrows():

    image_id = str(row["id"])
    url = str(row["image_url"])

    output_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

    # Skip already downloaded files
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        skipped += 1
        continue

    try:
        response = session.get(
            url,
            timeout=15,
            stream=True
        )

        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        downloaded += 1

        print(
            f"[{index + 1}/{total}] "
            f"Downloaded: {image_id}"
        )

    except Exception as e:
        failed += 1

        print(
            f"[{index + 1}/{total}] "
            f"FAILED: {image_id} | {e}"
        )

    time.sleep(0.2)

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(f"Downloaded: {downloaded}")
print(f"Skipped:    {skipped}")
print(f"Failed:     {failed}")
print(f"Total:      {total}")
print("=" * 60)