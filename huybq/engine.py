import os
import json
import math
import pickle
import urllib.request
from collections import Counter
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import calculate_recall_at_k, calculate_mrr_at_k
from utils.preprocess import clean_text

def load_corpus(filepath: str, cache_path: str) -> Dict[str, List[str]]:

    if os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    processed_corpus = {}
    print(f"Processing corpus from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Processing corpus"):
        row = json.loads(line)
        doc_id = str(row.get('_id', row.get('id', ''))) 
        
        # Gộp tiêu đề và nội dung
        title = row.get('title', '')
        text = row.get('text', '')
        full_content = f"{title}. {text}"
        
        # Tiền xử lý
        tokens = clean_text(full_content)
        processed_corpus[doc_id] = tokens
            
    print(f"Saving cached corpus to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(processed_corpus, f)
        
    print("Completed loading and processing the corpus!\n")
    return processed_corpus

def load_queries_and_qrels(
    queries_path: str, 
    qrels_path: str, 
    cache_path: str
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:

    if os.path.exists(cache_path):
        print(f"Loading cached queries & qrels from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    queries_tokens = {}
    ground_truths = {}
    
    # Xử lý queries
    print(f"Processing queries from {queries_path}...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing queries"):
            line = line.strip()
            if not line:
                continue
                
            row = json.loads(line)
            q_id = str(row.get('_id', row.get('id', '')))
            q_text = row.get('text', '')
            queries_tokens[q_id] = clean_text(q_text)

    # Xử lý qrels / ground truth
    print(f"Processing qrels from {qrels_path}...")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing qrels"):
            line = line.strip()
            if not line:
                continue
                
            row = json.loads(line)
            q_id = str(row.get('query-id', ''))
            c_id = str(row.get('corpus-id', ''))
            score = row.get('score', 0)
            
            # Điểm = 1 nghĩa là relevant
            if score == 1:
                if q_id not in ground_truths:
                    ground_truths[q_id] = []
                ground_truths[q_id].append(c_id)

    print(f"Saving cached queries & qrels to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump((queries_tokens, ground_truths), f)

    print("Completed loading queries and qrels!\n")
    return queries_tokens, ground_truths


class BM25Plus:
    """
    Implementation of BM25+ from scratch
    """
    def __init__(self, corpus: Dict[str, List[str]], k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        self.corpus_size = len(corpus)
        self.doc_lengths: Dict[str, int] = {}
        self.avgdl = 0.0
        
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.document_freqs: Dict[str, int] = Counter()
        self.idf: Dict[str, float] = {}
        
        self.initialize_index(corpus)
        self.compute_idf()

    def initialize_index(self, corpus: Dict[str, List[str]]):
        total_length = 0
        
        for doc_id, tokens in tqdm(corpus.items(), desc="Building BM25 Index"):
            length = len(tokens)
            self.doc_lengths[doc_id] = length
            total_length += length
            
            term_counts = Counter(tokens)
            
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count
                self.document_freqs[term] += 1
                
        if self.corpus_size > 0:
            self.avgdl = total_length / self.corpus_size

    def compute_idf(self):
        for term, df in self.document_freqs.items():
            numerator = self.corpus_size - df + 0.5
            denominator = df + 0.5
            self.idf[term] = math.log((numerator / denominator) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        scores = {doc_id: 0.0 for doc_id in self.doc_lengths.keys()}
        
        for term in query_tokens:
            if term not in self.inverted_index or term not in self.idf:
                continue
                
            idf_val = self.idf[term]
            
            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]
                
                length_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)
                tf_numerator = tf * (self.k1 + 1.0)
                tf_denominator = tf + self.k1 * length_norm
                
                term_score = idf_val * ((tf_numerator / tf_denominator) + self.delta)
                scores[doc_id] += term_score
                
        return scores

    def search(self, query_tokens: List[str], top_k: int = 10) -> List[str]:
        scores = self.get_scores(query_tokens)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]



def expand_query(bm25, corpus_tokens, query_tokens):
    if len(query_tokens) > 4:
        return query_tokens

    hits = bm25.search(query_tokens, top_k=7)
    if not hits:
        return query_tokens

    query_set = set(query_tokens)
    cand = Counter()
    for doc_id in hits:
        for t in set(corpus_tokens[doc_id]):
            if t not in query_set and t in bm25.idf:
                cand[t] += 1

    # IDF-rank các term có cross-doc agreement
    candidates = [(t, bm25.idf[t]) for t, f in cand.items() if f >= 2]
    if not candidates:
        return query_tokens

    best_term = max(candidates, key=lambda x: x[1])[0]
    return query_tokens * 3 + [best_term]


def main():
    # 1. Configuration
    BASE_URL = "https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn/resolve/main"
    CACHE_DIR = "cache"
    
    # URL trực tiếp từ HuggingFace
    CORPUS_URL = f"{BASE_URL}/corpus.jsonl"
    QUERIES_URL = f"{BASE_URL}/queries.jsonl"
    QRELS_URL = f"{BASE_URL}/qrels/train.jsonl"
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    CORPUS_FILE = os.path.join(CACHE_DIR, "corpus.jsonl")
    QUERIES_FILE = os.path.join(CACHE_DIR, "queries.jsonl")
    QRELS_FILE = os.path.join(CACHE_DIR, "train.jsonl")
    
    def download_file(url, local_path):
        if not os.path.exists(local_path):
            print(f"Downloading {url} to {local_path}...")
            urllib.request.urlretrieve(url, local_path)
            
    download_file(CORPUS_URL, CORPUS_FILE)
    download_file(QUERIES_URL, QUERIES_FILE)
    download_file(QRELS_URL, QRELS_FILE)
    
    CORPUS_CACHE = os.path.join(CACHE_DIR, "corpus_cache.pkl")
    QUERIES_CACHE = os.path.join(CACHE_DIR, "queries_cache.pkl")

    # Tham số đánh giá
    K_VALUES = [1, 5, 10, 20, 100]
    MAX_K = max(K_VALUES)

    # --- 2. Load và Tiền xử lý dữ liệu ---
    print("--- Đang tải & xử lý dữ liệu từ HuggingFace ---")
    
    # Tải Corpus
    corpus_tokens = load_corpus(
        filepath=CORPUS_FILE, 
        cache_path=CORPUS_CACHE
    )
    
    # Tải Queries và Ground Truth (qrels)
    queries_tokens, ground_truths = load_queries_and_qrels(
        queries_path=QUERIES_FILE,
        qrels_path=QRELS_FILE,
        cache_path=QUERIES_CACHE
    )

    # 3. Initialize BM25+ Engine
    print("Initializing BM25+ engine ...")
    bm25 = BM25Plus(corpus=corpus_tokens, k1=1.5, b=0.75, delta=1.0)
    print(f"Index built successfully! Vocabulary size: {len(bm25.inverted_index)} terms.")

    # 4. Evaluation Loop
    print(f"Evaluating BM25+ on train queries for K in {K_VALUES} ...")
    
    # Dictionary để lưu tổng điểm cho từng thang K
    total_recall = {k: 0.0 for k in K_VALUES}
    total_mrr = {k: 0.0 for k in K_VALUES}
    num_queries = len(queries_tokens)

    for q_id, q_tokens in tqdm(queries_tokens.items(), desc="Evaluating"):
        actual_docs = ground_truths.get(q_id, [])

        if len(actual_docs) > 0:
            # Mở rộng truy vấn bằng Pseudo-Relevance Feedback (PRF) tối ưu bằng TF-IDF
            expanded_q_tokens = expand_query(
                bm25=bm25,
                corpus_tokens=corpus_tokens,
                query_tokens=q_tokens
            )
            
            # Tìm kiếm với truy vấn đã được mở rộng
            retrieved_docs = bm25.search(expanded_q_tokens, top_k=MAX_K)

            # Tính toán metrics cho từng ngưỡng K
            for k in K_VALUES:
                total_recall[k] += calculate_recall_at_k(actual_docs, retrieved_docs, k)
                total_mrr[k] += calculate_mrr_at_k(actual_docs, retrieved_docs, k)
        else:
            # Trừ đi các query không có trong tập train qrels
            num_queries -= 1 

    # 5. Final Results
    print("\n" + "="*45)
    print("FINAL EVALUATION RESULTS")
    print("="*45)
    print(f"Total evaluated queries: {num_queries}")
    print("-" * 45)
    print(f"{'Metric':<15} | {'Recall':<10} | {'MRR':<10}")
    print("-" * 45)
    
    for k in K_VALUES:
        avg_recall = total_recall[k] / num_queries if num_queries > 0 else 0
        avg_mrr = total_mrr[k] / num_queries if num_queries > 0 else 0
        print(f"Top {k:<11} | {avg_recall:<10.4f} | {avg_mrr:<10.4f}")
        
    print("="*45)

if __name__ == "__main__":
    main()