"""
Lightweight In-Memory RAG (Retrieval-Augmented Generation) Service for DeepGuard.
Uses scikit-learn TF-IDF over passages from data/knowledge.json and data/library.json.
No external vector database and no external embedding service required.
100% local, offline, deterministic, and fast (<2ms).
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge.json"
LIBRARY_FILE = BASE_DIR / "data" / "library.json"


class LightweightRAG:
    _instance = None

    def __init__(self):
        self.passages = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_and_index(self):
        """Loads passages from knowledge.json and library.json and fits TF-IDF matrix."""
        raw_passages = []

        # 1. Load curated knowledge passages (2-4 sentences each)
        if KNOWLEDGE_FILE.exists():
            try:
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        text = item.get("text", "").strip()
                        source = item.get("source", "Help Hub")
                        if text:
                            raw_passages.append({
                                "id": item.get("id"),
                                "source": source,
                                "text": text
                            })
            except Exception as e:
                print("[!] Error loading knowledge.json:", e)

        # 2. Load case studies from library.json
        if LIBRARY_FILE.exists():
            try:
                with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for e in entries:
                        title = e.get("title", "Misinformation Case")
                        summary = e.get("summary", "")
                        tips = " ".join(e.get("how_to_spot", []))
                        text = f"{summary} Practical identification tips: {tips}".strip()
                        if text:
                            raw_passages.append({
                                "id": f"lib_{e.get('id', title)}",
                                "source": f"Misinformation Library: {title}",
                                "text": text
                            })
            except Exception as e:
                print("[!] Error loading library.json for RAG:", e)

        self.passages = raw_passages

        # Fit TF-IDF vectorizer if passages are present
        if self.passages:
            corpus = [f"{p['source']} {p['text']}" for p in self.passages]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=4000
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"[+] LightweightRAG initialized: indexed {len(self.passages)} passages into TF-IDF index.")
        else:
            print("[!] LightweightRAG warning: No passages found to index.")

    def _expand_query(self, query: str) -> str:
        """Expands short common user questions into effective search keywords."""
        q_norm = query.lower().strip().rstrip("?.! ")
        expansions = {
            "how it works": "how deepguard works multimodal pipeline 7 modules text image mismatch",
            "how does it work": "how deepguard works multimodal pipeline 7 modules text image mismatch",
            "how deepguard works": "how deepguard works multimodal pipeline 7 modules",
            "what is deepguard": "what is deepguard context-aware multimodal misinformation detection",
            "recycled images": "recycled image reverse image search spot misinformation",
            "how to spot recycled images": "recycled image reverse image search spot misinformation",
            "limitations": "limitations notice risk estimate semantic mismatch not factual proof",
            "what are the limitations": "limitations notice risk estimate semantic mismatch not factual proof",
            "what are the system limitations": "limitations notice risk estimate semantic mismatch not factual proof",
            "7 modules": "7 ai modules distilbert efficientnet exif vader viral clip shap",
            "modules": "7 ai modules distilbert efficientnet exif vader viral clip shap",
        }
        return expansions.get(q_norm, query)

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.04):
        """
        Retrieves top_k passages with cosine similarity >= min_similarity.
        Returns list of dicts: [{'source': ..., 'text': ..., 'score': ...}]
        """
        if self.vectorizer is None or self.tfidf_matrix is None or not query.strip():
            return []

        try:
            expanded_query = self._expand_query(query)
            q_vec = self.vectorizer.transform([expanded_query.strip()])
            scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

            ranked_indices = scores.argsort()[::-1]
            results = []

            for idx in ranked_indices:
                score = float(scores[idx])
                if score < min_similarity:
                    break
                p = self.passages[idx]
                results.append({
                    "id": p.get("id"),
                    "source": p["source"],
                    "text": p["text"],
                    "score": round(score, 3)
                })
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            print("RAG retrieval error:", e)
            return []

    def format_sources(self, retrieved_passages):
        """Builds a clean 'Based on: ...' attribution string."""
        if not retrieved_passages:
            return ""
        seen = []
        for p in retrieved_passages:
            src = p.get("source", "").strip()
            if src and src not in seen:
                seen.append(src)
        return "Based on: " + ", ".join(seen) if seen else ""
