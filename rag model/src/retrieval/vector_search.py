"""Vector search retrieval implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.indexing.vector_store import ChromaVectorStore


@dataclass
class RetrievalCandidate:
    """Represents a retrieved passage candidate across retrieval modalities."""
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    rank: int
    retrieval_mode: str
    raw_score: Optional[float] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_citation_str(self) -> str:
        """Format clean user-facing citation string."""
        return f"[{self.filename}, Page {self.page_number}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "raw_score": round(self.raw_score, 4) if self.raw_score is not None else None,
            "rank": self.rank,
            "retrieval_mode": self.retrieval_mode,
            "text": self.text,
            "citation": self.to_citation_str(),
            "extra_info": self.extra_info,
        }


class VectorSearcher:
    """Performs semantic vector search against the ChromaVectorStore."""

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalCandidate]:
        """
        Execute vector retrieval.

        Args:
            query: User search query.
            top_k: Number of candidates to retrieve.
            score_threshold: Optional minimum similarity threshold [0.0 - 1.0].

        Returns:
            Ranked list of RetrievalCandidate objects.
        """
        raw_results = self.vector_store.query(query_text=query, top_k=top_k)
        candidates: List[RetrievalCandidate] = []

        for rank, res in enumerate(raw_results, start=1):
            score = float(res.get("score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue

            candidate = RetrievalCandidate(
                chunk_id=res["chunk_id"],
                text=res["text"],
                doc_id=res["doc_id"],
                filename=res["filename"],
                page_number=res["page_number"],
                chunk_index=res["chunk_index"],
                score=score,
                raw_score=res.get("raw_distance"),
                rank=rank,
                retrieval_mode="vector",
            )
            candidates.append(candidate)

        return candidates
