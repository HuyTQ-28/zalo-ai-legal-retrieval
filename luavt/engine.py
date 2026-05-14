import math
import pickle
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from luavt.preprocess import load_corpus, load_queries, load_qrels
from utils.preprocess import clean_text
from utils.metrics import calculate_recall_at_k, calculate_mrr_at_k

def simple_tokenize(text: str) -> List[str]:
    return clean_text(text)

class TFIDFRetriever:
    def __init__(self):
        self.corpus_ids: List[str] = []
        
        # document frequency: {term: số lượng tài liệu chứa term}
        self.df: Dict[str, int] = defaultdict(int)
        
        # inverse document frequency: {term: idf}
        self.idf: Dict[str, float] = {}
        
        # inverted index: {term: {doc_idx: tf_idf_weight_normalized}}
        self.inverted_index: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        self.num_docs = 0

    def _compute_tf(self, count: int) -> float:
        return float(count)

    def fit(self, corpus_ids: List[str], corpus_texts: List[str], cache_dir: Path | None = None) -> None:
        """Xây dựng TF-IDF index từ corpus có hỗ trợ lưu cache."""
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / "tfidf_scratch_model.pkl"
            if cache_path.exists():
                print(f"Loading TF-IDF scratch model from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.corpus_ids = cached_data["corpus_ids"]
                    self.df = cached_data["df"]
                    self.idf = cached_data["idf"]
                    self.inverted_index = cached_data["inverted_index"]
                    self.num_docs = cached_data["num_docs"]
                return

        print("Fitting TF-IDF model from scratch...")
        self.corpus_ids = corpus_ids
        self.num_docs = len(corpus_texts)
        
        # Bước 1: Tính TF cục bộ và DF cho mỗi từ
        doc_term_counts: List[Dict[str, int]] = []
        for i, text in enumerate(corpus_texts):
            tokens = simple_tokenize(text)
            term_dict = defaultdict(int)
            for token in tokens:
                term_dict[token] += 1
            doc_term_counts.append(term_dict)
            
            for term in term_dict:
                self.df[term] += 1
                
        # Bước 2: Tính IDF = log(N / df(t))
        for term, df_val in self.df.items():
            self.idf[term] = math.log(self.num_docs / df_val)

        # Bước 3: Tính TF-IDF chuẩn hóa theo độ dài tài liệu và xây dựng Inverted Index
        for doc_idx, term_dict in enumerate(doc_term_counts):
            doc_len = sum(term_dict.values())
            if doc_len == 0:
                continue
            for term, count in term_dict.items():
                if term not in self.idf:
                    continue
                tf_val = self._compute_tf(count) / doc_len
                self.inverted_index[term][doc_idx] = tf_val * self.idf[term]
                    
        print(f"[TF-IDF from scratch] Index built: {self.num_docs:,} docs, {len(self.idf):,} terms")
        
        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "corpus_ids": self.corpus_ids,
                    "df": self.df,
                    "idf": self.idf,
                    "inverted_index": self.inverted_index,
                    "num_docs": self.num_docs
                }, f)

    def retrieve(self, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Truy xuất tài liệu sử dụng Cosine Similarity qua Inverted Index."""
        tokens = simple_tokenize(query_text)
        query_counts = defaultdict(int)
        for token in tokens:
            query_counts[token] += 1
            
        # Vector truy vấn — TF thô × IDF, không chuẩn hóa
        query_weights = {}
        query_len = sum(query_counts.values())
        for term, count in query_counts.items():
            if term in self.idf:
                tf_val = self._compute_tf(count) / query_len
                query_weights[term] = tf_val * self.idf[term]

        if not query_weights:
            return []

        # Tính điểm bằng dot product
        scores: Dict[int, float] = defaultdict(float)
        for term, q_weight in query_weights.items():
            if term in self.inverted_index: 
                for doc_idx, d_weight in self.inverted_index[term].items():
                    scores[doc_idx] += q_weight * d_weight
                    
        # Xếp hạng (sắp xếp giảm dần)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(self.corpus_ids[doc_idx], score) for doc_idx, score in sorted_docs]

    def retrieve_batch(self, query_ids: List[str], query_texts: List[str], top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        for qid, qtext in zip(query_ids, query_texts):
            results[qid] = self.retrieve(qtext, top_k)
        return results

def run_evaluation(data_dir: str | Path = None, top_k: int = 100, qrels_split: str = "test") -> dict:
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
    results = retriever.retrieve_batch(query_ids, query_texts, top_k=max(top_k, 100))

    def results_to_ranked_lists(results_dict):
        return {qid: [cid for cid, _ in ranked] for qid, ranked in results_dict.items()}

    ranked = results_to_ranked_lists(results)

    k_values = [k for k in [1, 5, 10, 15, 20] if k <= top_k]
    metrics = {}

    for k in k_values:
        recall_scores, mrr_scores = [], []
        for qid in query_ids:
            if qid not in qrels:
                continue
            relevant_docs = qrels[qid]
            retrieved = ranked.get(qid, [])
            recall_scores.append(calculate_recall_at_k(relevant_docs, retrieved, k))
            mrr_scores.append(calculate_mrr_at_k(relevant_docs, retrieved, k))
            
        metrics[f"recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        metrics[f"mrr@{k}"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    print(f"\n=== TF-IDF From Scratch ( TF × log(N/df)) ===")
    for k in k_values:
        print(f"  Recall@{k:<3} = {metrics[f'recall@{k}']:.4f}   MRR@{k:<3} = {metrics[f'mrr@{k}']:.4f}")

    return {"method": "tfidf_scratch", "top_k": top_k, **metrics}

if __name__ == "__main__":
    run_evaluation()
