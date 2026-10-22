"""Evaluation metrics for RAG retrieval, citations, and grounded answer quality."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.retrieval.vector_search import RetrievalCandidate


def compute_recall_at_k(
    retrieved_candidates: List[RetrievalCandidate],
    relevant_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Compute Recall@k: 1.0 if at least one labeled relevant chunk appears in top-k, else 0.0.
    """
    if not relevant_chunk_ids:
        return 0.0

    top_candidates = retrieved_candidates[:k]
    for c in top_candidates:
        if c.chunk_id in relevant_chunk_ids:
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    retrieved_candidates: List[RetrievalCandidate],
    relevant_chunk_ids: Set[str],
) -> float:
    """
    Compute Reciprocal Rank: 1 / rank of the first relevant candidate (1-based rank),
    or 0.0 if no relevant candidate is found.
    """
    if not relevant_chunk_ids:
        return 0.0

    for rank_idx, c in enumerate(retrieved_candidates, start=1):
        if c.chunk_id in relevant_chunk_ids:
            return 1.0 / rank_idx
    return 0.0


def compute_citation_correctness(
    cited_sources: List[str],
    expected_pages: Set[int],
    expected_filenames: Set[str],
) -> float:
    """
    Compute citation precision: proportion of cited [filename, Page X] references
    that actually match expected ground-truth page numbers and filenames.
    """
    if not cited_sources:
        return 0.0

    valid_citations = 0
    for citation in cited_sources:
        # Expected citation format: "[filename, Page X]"
        match = re.search(r"\[(.*?),\s*Page\s*(\d+)\]", citation, re.IGNORECASE)
        if match:
            fname = match.group(1).strip()
            pnum = int(match.group(2))
            if (not expected_filenames or fname in expected_filenames) and (
                not expected_pages or pnum in expected_pages
            ):
                valid_citations += 1

    return valid_citations / len(cited_sources)


def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level lexical F1 score between generated answer and reference answer.
    """
    pred_tokens = re.findall(r"\w+", prediction.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set.intersection(ref_set)
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    return (2 * precision * recall) / (precision + recall)
