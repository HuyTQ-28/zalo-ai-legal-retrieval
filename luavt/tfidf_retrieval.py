"""
TF-IDF Retrieval
Dùng TfidfVectorizer của scikit-learn + cosine similarity để xếp hạng tài liệu.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from luavt.preprocess import load_corpus, load_queries, load_qrels


class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),   
            min_df=2,
            max_df=0.85, 
            sublinear_tf=True,    
        )
        self.corpus_ids: list[str] = []
        self.corpus_matrix = None  

    def fit(self, corpus_ids: list[str], corpus_texts: list[str], cache_dir: Path | None = None) -> None:
        """Xây dựng TF-IDF index từ corpus (có lưu cache mô hình)."""
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / "tfidf_model.pkl"
            if cache_path.exists():
                print(f"Loading TF-IDF model from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.vectorizer = cached_data["vectorizer"]
                    self.corpus_ids = cached_data["corpus_ids"]
                    self.corpus_matrix = cached_data["corpus_matrix"]
                return

        print("Fitting TF-IDF model...")
        self.corpus_ids = corpus_ids
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        print(f"[TF-IDF] Index built: {self.corpus_matrix.shape[0]:,} docs, "
              f"{self.corpus_matrix.shape[1]:,} terms")
              
        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "corpus_ids": self.corpus_ids,
                    "corpus_matrix": self.corpus_matrix
                }, f)

    def retrieve(self, query_text: str, top_k: int = 100) -> list[tuple[str, float]]:
        q_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(q_vec, self.corpus_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus_ids[i], float(scores[i])) for i in top_indices]

    def retrieve_batch(
        self,
        query_ids: list[str],
        query_texts: list[str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Truy vấn toàn bộ queries, trả về dict {query_id: [(corpus_id, score)]}."""
        results = {}
        for qid, qtext in zip(query_ids, query_texts):
            results[qid] = self.retrieve(qtext, top_k)
        return results


def run_evaluation(
    data_dir: str | Path = None,
    top_k: int = 100,
    qrels_split: str = "test",
) -> dict:
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    data_dir = Path(data_dir)

    print("Đang tải dữ liệu...")
    corpus_ids, corpus_texts = load_corpus(data_dir / "corpus.jsonl")
    query_ids, query_texts = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels" / f"{qrels_split}.jsonl")

    cache_dir = data_dir.parent / ".cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    retriever = TFIDFRetriever()
    retriever.fit(corpus_ids, corpus_texts, cache_dir=cache_dir)

    print("Đang truy vấn...")
    results = retriever.retrieve_batch(query_ids, query_texts, top_k=top_k)

    from utils.metrics import evaluate, print_metrics
    ranked = results_to_ranked_lists(results)

    k_values = [k for k in [5, 10, 15, 20, 50] if k <= top_k]
    metrics = evaluate(ranked, qrels, k_values=k_values)
    print_metrics(metrics, method_name="TF-IDF (ngram 1-2, sublinear_tf)")

    return {"method": "tfidf", "top_k": top_k, **metrics}


def results_to_ranked_lists(
    results: dict[str, list[tuple[str, float]]]
) -> dict[str, list[str]]:
    """Chuyển {qid: [(cid, score)]} → {qid: [cid, ...]} (chỉ ids, đã sorted)."""
    return {qid: [cid for cid, _ in ranked] for qid, ranked in results.items()}


if __name__ == "__main__":
    run_evaluation()
