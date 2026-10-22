"""Reciprocal Rank Fusion (RRF) combining vector and BM25 rankings."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.retrieval.vector_search import RetrievalCandidate


class ReciprocalRankFusion:
    """
    Combines ranked candidate lists from vector search and BM25 lexical search
    using the standard Reciprocal Rank Fusion formula:
        RRF_score(d) = SUM_m [ 1 / (k + rank_m(d)) ]
    where k is the smoothing constant (default: 60) and rank_m is 1-based.
    """

    def __init__(self, rrf_k: int = 60):
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
        self.rrf_k = rrf_k

    def fuse(
        self,
        vector_candidates: List[RetrievalCandidate],
        bm25_candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Fuse vector and BM25 ranked candidate lists.

        Args:
            vector_candidates: Ranked list from vector search (1-based ranks).
            bm25_candidates: Ranked list from BM25 search (1-based ranks).
            top_k: Number of fused candidates to return.

        Returns:
            Ranked list of fused RetrievalCandidate objects with RRF score and provenance.
        """
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, RetrievalCandidate] = {}
        debug_ranks: Dict[str, Dict[str, Optional[int]]] = {}

        # Process vector rankings
        for cand in vector_candidates:
            cid = cand.chunk_id
            rank = cand.rank
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib
            candidate_map[cid] = cand
            if cid not in debug_ranks:
                debug_ranks[cid] = {"vector_rank": rank, "bm25_rank": None}
            else:
                debug_ranks[cid]["vector_rank"] = rank

        # Process BM25 rankings
        for cand in bm25_candidates:
            cid = cand.chunk_id
            rank = cand.rank
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib
            if cid not in candidate_map:
                candidate_map[cid] = cand
            if cid not in debug_ranks:
                debug_ranks[cid] = {"vector_rank": None, "bm25_rank": rank}
            else:
                debug_ranks[cid]["bm25_rank"] = rank

        # Sort descending by fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        actual_k = min(top_k, len(sorted_cids))
        fused_candidates: List[RetrievalCandidate] = []

        for final_rank, cid in enumerate(sorted_cids[:actual_k], start=1):
            base_cand = candidate_map[cid]
            ranks = debug_ranks[cid]
            fused_score = rrf_scores[cid]

            fused_cand = RetrievalCandidate(
                chunk_id=base_cand.chunk_id,
                text=base_cand.text,
                doc_id=base_cand.doc_id,
                filename=base_cand.filename,
                page_number=base_cand.page_number,
                chunk_index=base_cand.chunk_index,
                score=fused_score,
                raw_score=fused_score,
                rank=final_rank,
                retrieval_mode="hybrid_rrf",
                extra_info={
                    "vector_rank": ranks["vector_rank"],
                    "bm25_rank": ranks["bm25_rank"],
                    "rrf_score": fused_score,
                    "rrf_k": self.rrf_k,
                },
            )
            fused_candidates.append(fused_cand)

        return fused_candidates
