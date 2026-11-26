"""Basic retrieval quality evaluation utilities (Recall@k, MRR)."""

from __future__ import annotations

from rag_engine.pipeline.rag_pipeline import CardioRAGPipeline


def evaluate_retrieval(
    pipeline: CardioRAGPipeline,
    eval_cases: list[tuple[str, str]],
    *,
    top_k: int = 3,
) -> dict[str, float]:
    """Compute Recall@k and Mean Reciprocal Rank (MRR) on (query, expected_keyword) pairs."""
    if not eval_cases:
        return {"recall_at_k": 0.0, "mrr": 0.0}

    hits = 0
    rr_sum = 0.0
    for query, keyword in eval_cases:
        results = pipeline.query(query, top_k=top_k)
        keyword_l = keyword.lower()
        rank = None
        for idx, item in enumerate(results, start=1):
            if keyword_l in item["text"].lower() or keyword_l in item["source"].lower():
                rank = idx
                break
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank

    total = len(eval_cases)
    return {
        "recall_at_k": hits / total,
        "mrr": rr_sum / total,
    }
