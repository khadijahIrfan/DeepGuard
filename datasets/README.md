# DeepGuard Benchmark Datasets

This directory houses the benchmark and training datasets used to train, evaluate, and calibrate DeepGuard's multimodal detection models.

## Dataset Inventory

| Dataset | Modality | Samples | Description & Purpose |
|---|---|---|---|
| **WELFake** | Text | 72,134 claims | Large-scale balanced benchmark combining Kaggle, McIntire, Reuters, and BuzzFeed datasets for DistilBERT fine-tuning. |
| **LIAR** | Text | 12,836 statements | PolitiFact truth-o-meter statements categorized into 6-point truthfulness ratings. |
| **FakeNewsNet** | Text + Social | ~23,000 articles | Multi-dimensional dataset with BuzzFeed and PolitiFact news articles and social engagement graphs. |
| **COVID-19 Constraint** | Text | 10,700 posts | Health misinformation benchmark from Twitter and social platforms. |
| **FakeEdit** | Text + Edits | Multi-class | Fine-grained synthetic text manipulation and altered claim benchmark. |
| **Multimodal Mismatch** | Image + Caption | Benchmark set | Paired image-text context mismatch evaluation set located in `data/context_mismatch/images`. |

---

## Directory Organization

* **`text/COVID 19/`**: Constraint train, test, and validation sets.
* **`text/FakeEdit/`**: TSV partitioned benchmark splits.
* **`text/FakeNewsNet/`**: BuzzFeed & PolitiFact news content and user engagement data.
* **`text/LIAR/`**: TSV train, test, and validation files.
* **`text/WELFake_Dataset.csv`**: Uncompressed 245 MB master text corpus.
* **`text/zips_archive/`**: Preserved raw archive zip files.
