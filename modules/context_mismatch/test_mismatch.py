from modules.context_mismatch.mismatch import (
    clip_similarity,
    mismatch_score
)

image_path = r"E:\DeepGuard\backend\test_images\test.jpg"

text = "A cat sitting on a sofa"

print("=" * 50)

similarity = clip_similarity(
    image_path=image_path,
    text=text
)

print("Similarity:", similarity)

print("=" * 50)

result = mismatch_score(
    image_path=image_path,
    text=text
)

print(result)

print("=" * 50)