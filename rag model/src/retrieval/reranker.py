"""Optional CrossEncoder reranking module with lazy loading and graceful fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.vector_search import RetrievalCandidate

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved candidate passages using a compact CrossEncoder model.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, low RAM footprint ~350MB).
    Implements lazy loading to conserve memory until reranking is explicitly invoked.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._load_error: Optional[str] = None

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the CrossEncoder model when needed."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            logger.info("Lazy-loading CrossEncoder model: %s", self.model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:
            self._load_error = f"Failed to load CrossEncoder ({self.model_name}): {str(exc)}"
            logger.warning(self._load_error)
            return False

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        """
        Score question-passage pairs and reorder candidates.

        Args:
            query: User search query.
            candidates: Input candidates (e.g. from Hybrid RRF or Vector search).
            top_k: Number of reranked candidates to return.

        Returns:
            Tuple of (reranked_candidates, debug_info).
        """
        if not candidates or not self.enabled:
            debug_info = {
                "reranker_applied": False,
                "reason": "Disabled" if not self.enabled else "Empty candidates",
            }
            return candidates[:top_k] if top_k else candidates, debug_info

        if not self._ensure_model_loaded():
            logger.warning("Reranker unavailable (%s). Falling back to input ranking.", self._load_error)
            debug_info = {
                "reranker_applied": False,
                "reason": f"Fallback: {self._load_error}",
            }
            return candidates[:top_k] if top_k else candidates, debug_info

        # Prepare pairs for CrossEncoder
        pairs = [[query, c.text] for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("CrossEncoder prediction error: %s. Using original ranking.", exc)
            return candidates[:top_k] if top_k else candidates, {
                "reranker_applied": False,
                "reason": f"Prediction error: {str(exc)}",
            }

        # Pair each candidate with its CrossEncoder score
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            score_float = float(score)
            scored_candidates.append((score_float, cand))

        # Sort descending by CrossEncoder score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        cutoff = top_k if top_k is not None else len(scored_candidates)
        reranked_candidates: List[RetrievalCandidate] = []

        for new_rank, (ce_score, base_cand) in enumerate(scored_candidates[:cutoff], start=1):
            reranked_c = RetrievalCandidate(
                chunk_id=base_cand.chunk_id,
                text=base_cand.text,
                doc_id=base_cand.doc_id,
                filename=base_cand.filename,
                page_number=base_cand.page_number,
                chunk_index=base_cand.chunk_index,
                score=ce_score,
                raw_score=ce_score,
                rank=new_rank,
                retrieval_mode="hybrid_reranked",
                extra_info={
                    **base_cand.extra_info,
                    "previous_rank": base_cand.rank,
                    "cross_encoder_score": ce_score,
                    "reranker_model": self.model_name,
                },
            )
            reranked_candidates.append(reranked_c)

        debug_info = {
            "reranker_applied": True,
            "model_name": self.model_name,
            "candidates_scored": len(candidates),
        }
        return reranked_candidates, debug_info
