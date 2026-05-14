"""
Khởi tạo và cache các retrieval engine.
Được load một lần khi server start, tái sử dụng cho mọi request.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"

# ── Shared corpus store ──────────────────────────────────────────────────────

_corpus_raw: Optional[dict[str, dict]] = None   # doc_id → {title, text}
_bm25_qe_engine = None        # huybq — BM25+ với Query Expansion
_bm25_qe_corpus_tokens = None # corpus tokens gốc để dùng với expand_query
_bm25_engine = None           # huytq — BM25+ thuần
_tfidf_engine = None


def _load_corpus_raw() -> dict[str, dict]:
    """Load corpus.jsonl thành dict {doc_id: {title, text}}."""
    global _corpus_raw
    if _corpus_raw is not None:
        return _corpus_raw

    corpus_path = DATA_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {corpus_path}. "
            "Hãy chạy luavt/engine.py để tải dữ liệu."
        )

    _corpus_raw = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            doc_id = str(r.get("_id", r.get("id", "")))
            _corpus_raw[doc_id] = {
                "title": r.get("title", ""),
                "text": r.get("text", ""),
            }
    return _corpus_raw


# ── BM25+ loaders ─────────────────────────────────────────────────────────────

def get_bm25_qe():
    """Trả về (BM25Plus engine, corpus_tokens) của huybq, khởi tạo lần đầu nếu cần."""
    global _bm25_qe_engine, _bm25_qe_corpus_tokens
    if _bm25_qe_engine is not None:
        return _bm25_qe_engine, _bm25_qe_corpus_tokens

    from huybq.engine import BM25Plus, load_corpus as bm25_load_corpus

    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    corpus_cache = CACHE_DIR / "bm25_corpus_cache.pkl"

    _bm25_qe_corpus_tokens = bm25_load_corpus(
        filepath=str(DATA_DIR / "corpus.jsonl"),
        cache_path=str(corpus_cache),
    )
    _bm25_qe_engine = BM25Plus(corpus=_bm25_qe_corpus_tokens, k1=1.5, b=0.75, delta=1.0)
    return _bm25_qe_engine, _bm25_qe_corpus_tokens


def get_bm25():
    """Trả về BM25Plus engine của huytq (không có Query Expansion), khởi tạo lần đầu nếu cần."""
    global _bm25_engine
    if _bm25_engine is not None:
        return _bm25_engine

    from huytq.engine import BM25Plus, load_corpus as bm25_load_corpus

    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    # Dùng chung corpus cache vì cùng preprocessing (clean_text từ utils)
    corpus_cache = CACHE_DIR / "bm25_corpus_cache.pkl"

    corpus_tokens = bm25_load_corpus(
        filepath=str(DATA_DIR / "corpus.jsonl"),
        cache_path=str(corpus_cache),
    )
    _bm25_engine = BM25Plus(corpus=corpus_tokens, k1=1.5, b=0.75, delta=1.0)
    return _bm25_engine


# ── TF-IDF loader ─────────────────────────────────────────────────────────────

def get_tfidf():
    """Trả về TFIDFRetriever engine (luavt), khởi tạo lần đầu nếu cần."""
    global _tfidf_engine
    if _tfidf_engine is not None:
        return _tfidf_engine

    from luavt.engine import TFIDFRetriever
    from luavt.preprocess import load_corpus as tfidf_load_corpus

    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    corpus_ids, corpus_token_lists = tfidf_load_corpus(DATA_DIR / "corpus.jsonl")

    _tfidf_engine = TFIDFRetriever()
    _tfidf_engine.fit(corpus_ids, corpus_token_lists, cache_dir=CACHE_DIR)
    return _tfidf_engine


def get_corpus_raw() -> dict[str, dict]:
    return _load_corpus_raw()
