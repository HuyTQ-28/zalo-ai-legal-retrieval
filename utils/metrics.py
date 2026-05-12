from typing import List

def calculate_recall_at_k(actual_relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
    """
    Recall@K = (Number of relevant docs retrieved in Top K) / (Total relevant docs)
    """
    if not actual_relevant_docs:
        return 0.0
        
    top_k_retrieved = set(retrieved_docs[:k])
    relevant_set = set(actual_relevant_docs)
    
    hits = len(top_k_retrieved.intersection(relevant_set))
    
    return hits / len(relevant_set)

def calculate_mrr_at_k(actual_relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank @ K.
    Returns 1 / (Rank of first relevant document). Returns 0.0 if none found in Top K
    """
    if not actual_relevant_docs:
        return 0.0
        
    relevant_set = set(actual_relevant_docs)
    
    for rank, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_set:
            return 1.0 / (rank + 1)
            
    return 0.0