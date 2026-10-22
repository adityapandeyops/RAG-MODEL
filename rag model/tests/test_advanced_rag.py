"""Unit tests for Milestones 2, 3, 4, and 5: BM25, RRF, Reranker, and Metrics."""

import pytest

from src.evaluation.metrics import (
    compute_citation_correctness,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_token_f1,
)
from src.ingestion.chunking import ChunkMetadata
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_search import RetrievalCandidate


class TestBM25Searcher:
    @pytest.fixture
    def sample_chunks(self):
        return [
            ChunkMetadata(
                chunk_id="doc1_p1_c0",
                doc_id="doc1",
                filename="rag_paper.pdf",
                page_number=1,
                chunk_index=0,
                text="Dense vector embeddings excel at semantic search but struggle with rare keywords and acronyms like GPU or ASIC.",
                char_count=110,
                token_estimate=15,
            ),
            ChunkMetadata(
                chunk_id="doc1_p2_c1",
                doc_id="doc1",
                filename="rag_paper.pdf",
                page_number=2,
                chunk_index=1,
                text="BM25 is a probabilistic lexical retrieval algorithm based on term frequency and document length normalization.",
                char_count=111,
                token_estimate=15,
            ),
            ChunkMetadata(
                chunk_id="doc2_p1_c0",
                doc_id="doc2",
                filename="bio_paper.pdf",
                page_number=1,
                chunk_index=0,
                text="Cellular respiration in eukaryotic mitochondria converts oxygen and glucose into adenosine triphosphate.",
                char_count=104,
                token_estimate=14,
            ),
        ]

    def test_bm25_exact_keyword_matching(self, sample_chunks):
        searcher = BM25Searcher(sample_chunks)
        results = searcher.search("ASIC and GPU acronyms", top_k=2)
        assert len(results) >= 1
        assert results[0].chunk_id == "doc1_p1_c0"
        assert results[0].filename == "rag_paper.pdf"
        assert results[0].page_number == 1
        assert results[0].retrieval_mode == "bm25"
        assert results[0].rank == 1

    def test_bm25_empty_query(self, sample_chunks):
        searcher = BM25Searcher(sample_chunks)
        assert searcher.search("", top_k=5) == []


class TestReciprocalRankFusion:
    def test_rrf_formula_and_merging(self):
        rrf = ReciprocalRankFusion(rrf_k=60)

        # Doc A: rank 1 in vector, rank 2 in BM25
        # Doc B: rank 2 in vector, not in BM25
        # Doc C: not in vector, rank 1 in BM25
        vec_cands = [
            RetrievalCandidate(
                chunk_id="docA",
                text="Text A",
                doc_id="d1",
                filename="f1.pdf",
                page_number=1,
                chunk_index=0,
                score=0.9,
                rank=1,
                retrieval_mode="vector",
            ),
            RetrievalCandidate(
                chunk_id="docB",
                text="Text B",
                doc_id="d2",
                filename="f2.pdf",
                page_number=2,
                chunk_index=0,
                score=0.7,
                rank=2,
                retrieval_mode="vector",
            ),
        ]

        bm25_cands = [
            RetrievalCandidate(
                chunk_id="docC",
                text="Text C",
                doc_id="d3",
                filename="f3.pdf",
                page_number=3,
                chunk_index=0,
                score=0.8,
                rank=1,
                retrieval_mode="bm25",
            ),
            RetrievalCandidate(
                chunk_id="docA",
                text="Text A",
                doc_id="d1",
                filename="f1.pdf",
                page_number=1,
                chunk_index=0,
                score=0.6,
                rank=2,
                retrieval_mode="bm25",
            ),
        ]

        fused = rrf.fuse(vec_cands, bm25_cands, top_k=3)
        assert len(fused) == 3

        # Doc A should be rank 1 because it has rank 1 in vector (1/61) + rank 2 in BM25 (1/62)
        expected_score_a = (1.0 / 61.0) + (1.0 / 62.0)
        assert fused[0].chunk_id == "docA"
        assert pytest.approx(fused[0].score, 0.0001) == expected_score_a
        assert fused[0].rank == 1
        assert fused[0].retrieval_mode == "hybrid_rrf"
        assert fused[0].extra_info["vector_rank"] == 1
        assert fused[0].extra_info["bm25_rank"] == 2


class TestCrossEncoderReranker:
    def test_reranker_disabled_passthrough(self):
        reranker = CrossEncoderReranker(enabled=False)
        candidates = [
            RetrievalCandidate(
                chunk_id="c1",
                text="First text",
                doc_id="d1",
                filename="f1.pdf",
                page_number=1,
                chunk_index=0,
                score=0.5,
                rank=1,
                retrieval_mode="hybrid_rrf",
            )
        ]
        out, debug = reranker.rerank("some query", candidates)
        assert not debug["reranker_applied"]
        assert len(out) == 1
        assert out[0].chunk_id == "c1"


class TestEvaluationMetrics:
    def test_recall_at_k(self):
        candidates = [
            RetrievalCandidate(
                chunk_id="c1",
                text="",
                doc_id="",
                filename="",
                page_number=1,
                chunk_index=0,
                score=0.9,
                rank=1,
                retrieval_mode="vector",
            ),
            RetrievalCandidate(
                chunk_id="c2",
                text="",
                doc_id="",
                filename="",
                page_number=1,
                chunk_index=1,
                score=0.8,
                rank=2,
                retrieval_mode="vector",
            ),
        ]
        assert compute_recall_at_k(candidates, {"c1"}, k=1) == 1.0
        assert compute_recall_at_k(candidates, {"c2"}, k=1) == 0.0
        assert compute_recall_at_k(candidates, {"c2"}, k=2) == 1.0
        assert compute_recall_at_k(candidates, {"c3"}, k=5) == 0.0

    def test_reciprocal_rank(self):
        candidates = [
            RetrievalCandidate(
                chunk_id="c1",
                text="",
                doc_id="",
                filename="",
                page_number=1,
                chunk_index=0,
                score=0.9,
                rank=1,
                retrieval_mode="vector",
            ),
            RetrievalCandidate(
                chunk_id="c2",
                text="",
                doc_id="",
                filename="",
                page_number=1,
                chunk_index=1,
                score=0.8,
                rank=2,
                retrieval_mode="vector",
            ),
        ]
        assert compute_reciprocal_rank(candidates, {"c1"}) == 1.0
        assert compute_reciprocal_rank(candidates, {"c2"}) == 0.5
        assert compute_reciprocal_rank(candidates, {"c99"}) == 0.0

    def test_citation_correctness(self):
        citations = ["[paper.pdf, Page 2]", "[wrong.pdf, Page 10]"]
        score = compute_citation_correctness(
            cited_sources=citations,
            expected_pages={2},
            expected_filenames={"paper.pdf"},
        )
        assert score == 0.5

    def test_token_f1(self):
        pred = "Photosynthesis generates oxygen in chloroplasts"
        ref = "Chloroplasts produce oxygen through photosynthesis"
        f1 = compute_token_f1(pred, ref)
        assert f1 >= 0.6
