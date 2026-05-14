import json
import pickle
from pathlib import Path
from utils.preprocess import clean_text


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_corpus(corpus_path: str | Path, force_preprocess: bool = False) -> tuple[list[str], list[list[str]]]:
    """Trả về (ids, token_lists) đã tiền xử lý từ corpus.jsonl, có lưu cache."""
    corpus_path = Path(corpus_path)
    cache_path = corpus_path.parent.parent / ".cache" / f"{corpus_path.stem}_tokens.pkl"

    if not force_preprocess and cache_path.exists():
        print(f"Loading corpus from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Preprocessing corpus: {corpus_path}")
    records = load_jsonl(corpus_path)
    ids, token_lists = [], []
    for r in records:
        ids.append(r["_id"])
        combined = (r.get("title", "") + " " + r.get("text", "")).strip()
        token_lists.append(clean_text(combined))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((ids, token_lists), f)

    return ids, token_lists


def load_queries(queries_path: str | Path, force_preprocess: bool = False) -> tuple[list[str], list[str]]:
    """Trả về (ids, texts) thô từ queries.jsonl để engine tự tokenize khi retrieve."""
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
        texts.append(r.get("text", ""))

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
