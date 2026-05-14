"""
Hàm search thống nhất cho cả BM25+ và TF-IDF.
Trả về list[dict] với các trường: doc_id, score, title, snippet.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SNIPPET_LEN = 300  # ký tự trích đoạn


def _make_snippet(text: str, length: int = SNIPPET_LEN) -> str:
    text = text.strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."


def _bm25_search_with_scores(bm25, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
    """Gọi get_scores() một lần duy nhất, tránh gọi lại lần 2 để lấy score."""
    scores = bm25.get_scores(query_tokens)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


def search_bm25_qe(query: str, top_k: int = 10) -> list[dict]:
    """BM25+ với Query Expansion (huybq)."""
    from api.loader import get_bm25_qe, get_corpus_raw
    from huybq.engine import expand_query
    from utils.preprocess import clean_text

    bm25, corpus_tokens = get_bm25_qe()
    corpus_raw = get_corpus_raw()

    query_tokens = clean_text(query)
    expanded = expand_query(bm25, corpus_tokens, query_tokens)
    hits = _bm25_search_with_scores(bm25, expanded, top_k)

    results = []
    for doc_id, score in hits:
        raw = corpus_raw.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "title": raw.get("title", ""),
            "snippet": _make_snippet(raw.get("text", "")),
        })
    return results


def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    """BM25+ thuần, không có Query Expansion (huytq)."""
    from api.loader import get_bm25, get_corpus_raw
    from utils.preprocess import clean_text

    bm25 = get_bm25()
    corpus_raw = get_corpus_raw()

    query_tokens = clean_text(query)
    hits = _bm25_search_with_scores(bm25, query_tokens, top_k)

    results = []
    for doc_id, score in hits:
        raw = corpus_raw.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "title": raw.get("title", ""),
            "snippet": _make_snippet(raw.get("text", "")),
        })
    return results


def search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    from api.loader import get_tfidf, get_corpus_raw

    tfidf = get_tfidf()
    corpus_raw = get_corpus_raw()

    hits = tfidf.retrieve(query, top_k=top_k)

    results = []
    for doc_id, score in hits:
        raw = corpus_raw.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "title": raw.get("title", ""),
            "snippet": _make_snippet(raw.get("text", "")),
        })
    return results


def search(query: str, method: str = "bm25_qe", top_k: int = 10) -> list[dict]:
    """
    method: "bm25_qe" | "bm25" | "tfidf"
      - bm25_qe : BM25+ với Query Expansion (huybq)
      - bm25     : BM25+ thuần (huytq)
      - tfidf    : TF-IDF (luavt)
    Trả về list[dict] với các trường: doc_id, score, title, snippet.
    """
    query = query.strip()
    if not query:
        return []

    if method == "tfidf":
        return search_tfidf(query, top_k)
    if method == "bm25":
        return search_bm25(query, top_k)
    return search_bm25_qe(query, top_k)
