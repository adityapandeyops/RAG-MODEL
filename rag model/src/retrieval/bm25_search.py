"""BM25 Lexical Keyword Search implementation using rank-bm25."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.ingestion.chunking import ChunkMetadata
from src.retrieval.vector_search import RetrievalCandidate


def simple_tokenizer(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Searcher:
    """Performs BM25 keyword retrieval over indexed document chunks."""

    def __init__(self, chunks: Optional[List[ChunkMetadata]] = None):
        self.chunks: List[ChunkMetadata] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.fit(chunks)

    def fit(self, chunks: List[ChunkMetadata]) -> None:
        """Fit BM25 index on the supplied chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [simple_tokenizer(c.text) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def add_chunks(self, new_chunks: List[ChunkMetadata]) -> None:
        """Add new chunks and re-index the BM25 corpus."""
        existing_ids = {c.chunk_id for c in self.chunks}
        filtered_new = [c for c in new_chunks if c.chunk_id not in existing_ids]
        if not filtered_new:
            return
        all_chunks = self.chunks + filtered_new
        self.fit(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalCandidate]:
        """
        Execute BM25 keyword search.

        Args:
            query: Keyword query string.
            top_k: Number of top documents to return.
            min_score: Minimum BM25 score cutoff.

        Returns:
            Ranked list of RetrievalCandidate objects.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokens = simple_tokenizer(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)

        # Pair scores with chunk objects
        scored_pairs = []
        for idx, score in enumerate(raw_scores):
            if score > min_score:
                scored_pairs.append((score, self.chunks[idx]))

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        actual_k = min(top_k, len(scored_pairs))
        candidates: List[RetrievalCandidate] = []

        # Find max score for relative scaling
        max_score = scored_pairs[0][0] if scored_pairs else 1.0
        if max_score <= 0:
            max_score = 1.0

        for rank, (score, chunk) in enumerate(scored_pairs[:actual_k], start=1):
            normalized_score = min(1.0, score / max_score)
            candidate = RetrievalCandidate(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=normalized_score,
                raw_score=score,
                rank=rank,
                retrieval_mode="bm25",
            )
            candidates.append(candidate)

        return candidates
