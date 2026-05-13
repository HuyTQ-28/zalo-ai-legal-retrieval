import json
import re
import unicodedata
import pickle
from pathlib import Path
from pyvi import ViTokenizer
HAS_PYVI = True

STOPWORDS = {
    "và", "của", "là", "có", "trong", "được", "các", "cho", "với", "về",
    "không", "này", "đó", "khi", "từ", "theo", "đến", "tại", "hoặc", "để",
    "những", "một", "như", "bằng", "thì", "mà", "vào", "ra", "còn", "cũng",
    "đã", "sẽ", "đang", "bị", "do", "vì", "nên", "nếu", "thế", "cái",
    "đây", "ở", "lên", "xuống", "hay", "cùng", "sau", "trước", "trên", "dưới",
    "bởi", "qua", "lại", "nữa", "rằng", "thì", "mỗi", "giữa", "chỉ",
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa unicode về dạng NFC, chuyển thường, xóa ký tự đặc biệt."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Giữ lại chữ cái tiếng Việt, số, khoảng trắng
    text = re.sub(r"[^\w\s/.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tách từ bằng PyVi nếu có, ngược lại tách theo khoảng trắng."""
    if HAS_PYVI:
        text = ViTokenizer.tokenize(text)
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str, remove_sw: bool = True) -> list[str]:
    """Pipeline tiền xử lý hoàn chỉnh: chuẩn hóa → tách từ → (xóa stopword)."""
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    # if remove_sw:
    #     tokens = remove_stopwords(tokens)
    return tokens


def preprocess_to_string(text: str, remove_sw: bool = True) -> str:
    """Trả về chuỗi sau tiền xử lý (dùng cho TF-IDF vectorizer)."""
    return " ".join(preprocess(text, remove_sw))


# ── Load dữ liệu ──────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_corpus(corpus_path: str | Path, force_preprocess: bool = False) -> tuple[list[str], list[str]]:
    """Trả về (ids, texts) đã tiền xử lý từ corpus.jsonl, có lưu cache."""
    corpus_path = Path(corpus_path)
    cache_path = corpus_path.parent.parent / ".cache" / f"{corpus_path.stem}_processed.pkl"
    
    if not force_preprocess and cache_path.exists():
        print(f"Loading corpus from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Preprocessing corpus: {corpus_path}")
    records = load_jsonl(corpus_path)
    ids, texts = [], []
    for r in records:
        ids.append(r["_id"])
        combined = (r.get("title", "") + " " + r.get("text", "")).strip()
        texts.append(preprocess_to_string(combined))
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((ids, texts), f)
        
    return ids, texts


def load_queries(queries_path: str | Path, force_preprocess: bool = False) -> tuple[list[str], list[str]]:
    """Trả về (ids, texts) đã tiền xử lý từ queries.jsonl, có lưu cache."""
    queries_path = Path(queries_path)
    cache_path = queries_path.parent.parent / ".cache" / f"{queries_path.stem}_processed.pkl"
    
    if not force_preprocess and cache_path.exists():
        print(f"Loading queries from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Preprocessing queries: {queries_path}")
    records = load_jsonl(queries_path)
    ids, texts = [], []
    for r in records:
        ids.append(r["_id"])
        texts.append(preprocess_to_string(r.get("text", "")))
        
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((ids, texts), f)
        
    return ids, texts


def load_qrels(qrels_path: str | Path) -> dict[str, list[str]]:
    """Trả về dict {query_id: [corpus_id, ...]} từ file qrels.jsonl."""
    records = load_jsonl(qrels_path)
    qrels: dict[str, list[str]] = {}
    for r in records:
        qid = r["query-id"]
        cid = r["corpus-id"]
        qrels.setdefault(qid, []).append(cid)
    return qrels


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data"
    corpus_ids, corpus_texts = load_corpus(base / "corpus.jsonl")
    query_ids, query_texts = load_queries(base / "queries.jsonl")
    qrels = load_qrels(base / "qrels" / "test.jsonl")

    print(f"Corpus  : {len(corpus_ids):,} documents")
    print(f"Queries : {len(query_ids):,} queries")
    print(f"Qrels   : {len(qrels):,} query-doc pairs")
    print("\nVí dụ corpus[0]:")
    print(f"  ID  : {corpus_ids[0]}")
    print(f"  Text: {corpus_texts[0][:120]}...")
    print("\nVí dụ query[0]:")
    print(f"  ID  : {query_ids[0]}")
    print(f"  Text: {query_texts[0]}")
