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

class TFIDFRetriever:
    def __init__(self):
        # document frequency: {term: số lượng tài liệu chứa term}
        self.df: Dict[str, int] = defaultdict(int)
        
        # inverse document frequency: {term: idf}
        self.idf: Dict[str, float] = {}
        
        # inverted index: {term: {doc_id: tf_idf_weight}}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        self.num_docs = 0

    def fit(self, corpus_ids: List[str], corpus_token_lists: List[List[str]], cache_dir: Path | None = None) -> None:
        """Xây dựng TF-IDF index từ corpus có hỗ trợ lưu cache."""
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / "tfidf_scratch_model.pkl"
            if cache_path.exists():
                print(f"Loading TF-IDF scratch model from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.df = cached_data["df"]
                    self.idf = cached_data["idf"]
                    self.inverted_index = cached_data["inverted_index"]
                    self.num_docs = cached_data["num_docs"] 
                return

        print("Fitting TF-IDF model from scratch...")
        self.num_docs = len(corpus_token_lists)

        # Bước 1: Tính TF và DF cho mỗi từ
        doc_term_counts: List[Dict[str, int]] = []
        for tokens in corpus_token_lists:
            term_dict = defaultdict(int)
            for token in tokens:
                term_dict[token] += 1
            doc_term_counts.append(term_dict)
            
            for term in term_dict:
                self.df[term] += 1 # df là số lượng tài liệu chứa term
                
        # Bước 2: Tính IDF = log(N / df(t))
        for term, df_val in self.df.items():
            self.idf[term] = math.log(self.num_docs / df_val)

        # Bước 3: Tính TF-IDF chuẩn hóa theo độ dài tài liệu và xây dựng Inverted Index
        for doc_id, term_dict in zip(corpus_ids, doc_term_counts):
            doc_len = sum(term_dict.values())
            if doc_len == 0:
                continue
            for term, count in term_dict.items():
                if term not in self.idf:
                    continue
                tf_val = count / doc_len
                self.inverted_index[term][doc_id] = tf_val * self.idf[term]
                    
        print(f"[TF-IDF from scratch] Index built: {self.num_docs:,} docs, {len(self.idf):,} terms")
        
        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "df": self.df,
                    "idf": self.idf,
                    "inverted_index": self.inverted_index,
                    "num_docs": self.num_docs
                }, f)

    def retrieve(self, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
        tokens = clean_text(query_text)
        query_counts = defaultdict(int)
        for token in tokens:
            query_counts[token] += 1
            
        # Vector truy vấn — TF × IDF
        query_weights = {}
        query_len = sum(query_counts.values())
        for term, count in query_counts.items():
            if term in self.idf:
                tf_val = count / query_len
                query_weights[term] = tf_val * self.idf[term]

        if not query_weights:
            return []

        # dot product = sum(q_weight * d_weight) cho tất cả các term chung giữa truy vấn và tài liệu
        scores: Dict[str, float] = defaultdict(float)
        for term, q_weight in query_weights.items():
            if term in self.inverted_index:
                for doc_id, d_weight in self.inverted_index[term].items():
                    scores[doc_id] += q_weight * d_weight

        # Xếp hạng (sắp xếp giảm dần)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return list(sorted_docs)

def run_evaluation(data_dir: str | Path = None, top_k: int = 100, qrels_split: str = "train") -> dict:
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    data_dir = Path(data_dir)

    print("Đang tải dữ liệu...")
    corpus_ids, corpus_token_lists = load_corpus(data_dir / "corpus.jsonl")
    query_ids, query_texts = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels" / f"{qrels_split}.jsonl")

    cache_dir = data_dir.parent / ".cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    retriever = TFIDFRetriever()
    retriever.fit(corpus_ids, corpus_token_lists, cache_dir=cache_dir)

    qid_to_text = dict(zip(query_ids, query_texts))
    k_values = [k for k in [1, 5, 10, 15, 20] if k <= top_k]
    total_recall = {k: 0.0 for k in k_values}
    total_mrr = {k: 0.0 for k in k_values}
    num_queries = 0

    print("Đang truy vấn...")
    for qid, relevant_docs in qrels.items():
        if qid not in qid_to_text:
            continue
        retrieved = [doc_id for doc_id, _ in retriever.retrieve(qid_to_text[qid], top_k=max(top_k, 100))]
        num_queries += 1
        for k in k_values:
            total_recall[k] += calculate_recall_at_k(relevant_docs, retrieved, k)
            total_mrr[k] += calculate_mrr_at_k(relevant_docs, retrieved, k)

    metrics = {}
    print(f"\n=== TF-IDF From Scratch ( TF × log(N/df)) ===")
    for k in k_values:
        metrics[f"recall@{k}"] = total_recall[k] / num_queries if num_queries > 0 else 0.0
        metrics[f"mrr@{k}"] = total_mrr[k] / num_queries if num_queries > 0 else 0.0
        print(f"  Recall@{k:<3} = {metrics[f'recall@{k}']:.4f}   MRR@{k:<3} = {metrics[f'mrr@{k}']:.4f}")

    return {"method": "tfidf_scratch", "top_k": top_k, **metrics}

if __name__ == "__main__":
    run_evaluation()
