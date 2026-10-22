"""Unit tests for Milestone 1: Ingestion, Chunking, Embeddings, Vector Store, and Grounding."""

import io
import shutil
import tempfile
from pathlib import Path

import pypdf
import pytest

from src.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    AnswerGenerator,
)
from src.indexing.embeddings import (
    DeterministicFallbackEmbedder,
    OllamaEmbedder,
)
from src.indexing.vector_store import ChromaVectorStore
from src.ingestion.chunking import ChunkMetadata, ParagraphAwareChunker
from src.ingestion.pdf_loader import DocumentMetadata, PageText, PDFLoader
from src.retrieval.vector_search import RetrievalCandidate, VectorSearcher


def create_synthetic_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory valid PDF for testing."""
    writer = pypdf.PdfWriter()
    for text in pages_text:
        # Create a blank page and add annotations or text
        # In pypdf, we can create a blank page:
        page = writer.add_blank_page(width=300, height=300)
        # Note: blank page has no extractable text unless text operators are added.
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


class TestPDFLoader:
    def test_missing_file(self):
        loader = PDFLoader()
        doc = loader.load_pdf("non_existent_file.pdf")
        assert doc.extraction_error is not None
        assert "not found" in doc.extraction_error.lower()

    def test_empty_file(self):
        loader = PDFLoader()
        doc = loader.load_pdf(b"", filename="empty.pdf")
        assert doc.extraction_error is not None
        assert "0 bytes" in doc.extraction_error

    def test_corrupt_file(self):
        loader = PDFLoader()
        doc = loader.load_pdf(b"not a valid pdf header", filename="corrupt.pdf")
        assert doc.extraction_error is not None
        assert "Failed to parse PDF" in doc.extraction_error

    def test_synthetic_blank_page_detection(self):
        loader = PDFLoader(low_text_threshold=50)
        pdf_bytes = create_synthetic_pdf(["", ""])
        doc = loader.load_pdf(pdf_bytes, filename="blank_sample.pdf")
        assert doc.extraction_error is None
        assert doc.total_pages == 2
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[0].extraction_status == "empty_page"
        assert doc.pages[1].page_number == 2

    def test_real_pdf_page_aware_extraction(self):
        sample_path = Path("data/sample_research_paper.pdf")
        if sample_path.exists():
            loader = PDFLoader()
            doc = loader.load_pdf(sample_path)
            assert doc.extraction_error is None
            assert doc.total_pages == 4
            assert len(doc.pages) == 4
            assert doc.pages[0].page_number == 1
            assert doc.pages[0].extraction_status == "ok"
            assert "Retrieval-Augmented Generation" in doc.pages[0].text
            assert "BM25" in doc.pages[1].text
            assert "Cross-Encoder" in doc.pages[2].text
            assert "Citation Provenance" in doc.pages[3].text


class TestChunker:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="chunk_size must be > 50"):
            ParagraphAwareChunker(chunk_size=40)
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            ParagraphAwareChunker(chunk_size=100, chunk_overlap=120)

    def test_paragraph_chunking_metadata(self):
        chunker = ParagraphAwareChunker(chunk_size=200, chunk_overlap=50)
        page = PageText(
            doc_id="doc_abc",
            filename="paper.pdf",
            page_number=3,
            text=(
                "First paragraph describes the experimental setup in detail.\n\n"
                "Second paragraph presents the empirical findings across five models.\n\n"
                "Third paragraph concludes with implications for future research."
            ),
            char_count=180,
            extraction_status="ok",
        )
        chunks = chunker.chunk_page(page)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.doc_id == "doc_abc"
            assert c.filename == "paper.pdf"
            assert c.page_number == 3
            assert c.chunk_id.startswith("doc_abc_p3_c")
            assert c.char_count > 0
            assert c.token_estimate >= 1

    def test_empty_page_chunking(self):
        chunker = ParagraphAwareChunker()
        page = PageText(
            doc_id="doc_empty",
            filename="empty.pdf",
            page_number=1,
            text="",
            char_count=0,
            extraction_status="empty_page",
        )
        chunks = chunker.chunk_page(page)
        assert chunks == []


class TestEmbeddings:
    def test_deterministic_fallback_embedder(self):
        embedder = DeterministicFallbackEmbedder(dimension=384)
        assert embedder.get_dimension() == 384
        vec = embedder.embed_query("Quantum computing and neural networks")
        assert len(vec) == 384
        # Verify normalization
        magnitude = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(magnitude, 0.01) == 1.0

    def test_deterministic_consistent_vectors(self):
        embedder = DeterministicFallbackEmbedder(dimension=384)
        vec1 = embedder.embed_query("Local RAG Research Assistant")
        vec2 = embedder.embed_query("Local RAG Research Assistant")
        assert vec1 == vec2

    def test_ollama_offline_health_check(self):
        # Point to port that is offline
        embedder = OllamaEmbedder(base_url="http://localhost:59999")
        healthy, msg, diag = embedder.check_health()
        assert not healthy
        assert diag.error_code in ("NOT_INSTALLED", "SERVER_STOPPED", "CONNECTION_REFUSED", "TIMEOUT")
        assert len(diag.action_hint) > 0

    def test_dimension_mismatch_prevention(self):
        import tempfile, shutil
        temp_dir = tempfile.mkdtemp()
        try:
            embedder_384 = DeterministicFallbackEmbedder(dimension=384)
            store = ChromaVectorStore(persist_directory=temp_dir, collection_name="test_dim", embedder=embedder_384)
            chunk = ChunkMetadata(
                chunk_id="c1", doc_id="d1", filename="f.pdf", page_number=1, chunk_index=0,
                text="sample text", char_count=11, token_estimate=2
            )
            store.index_chunks([chunk])

            # Now create a store on the same directory with a 768-dim embedder
            embedder_768 = DeterministicFallbackEmbedder(dimension=768)
            store_incompatible = ChromaVectorStore(persist_directory=temp_dir, collection_name="test_dim", embedder=embedder_768)
            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                store_incompatible.index_chunks([chunk], force=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestChromaVectorStore:
    @pytest.fixture
    def temp_store(self):
        temp_dir = tempfile.mkdtemp()
        embedder = DeterministicFallbackEmbedder(dimension=384)
        store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="test_collection",
            embedder=embedder,
        )
        yield store
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_index_and_query(self, temp_store):
        chunks = [
            ChunkMetadata(
                chunk_id="doc1_p1_c0",
                doc_id="doc1",
                filename="biology.pdf",
                page_number=1,
                chunk_index=0,
                text="Mitochondria are the powerhouses of eukaryotic cells.",
                char_count=52,
                token_estimate=8,
            ),
            ChunkMetadata(
                chunk_id="doc1_p2_c1",
                doc_id="doc1",
                filename="biology.pdf",
                page_number=2,
                chunk_index=1,
                text="Photosynthesis converts solar energy into chemical energy in chloroplasts.",
                char_count=73,
                token_estimate=10,
            ),
        ]
        indexed_count = temp_store.index_chunks(chunks)
        assert indexed_count == 2
        assert temp_store.is_document_indexed("doc1")

        # Deduplication test: re-indexing unchanged doc should skip
        reindexed_count = temp_store.index_chunks(chunks)
        assert reindexed_count == 0

        # Query
        results = temp_store.query("Where does photosynthesis happen?", top_k=2)
        assert len(results) == 2
        assert results[0]["filename"] == "biology.pdf"
        assert "page_number" in results[0]
        assert results[0]["retrieval_mode"] == "vector"

    def test_vector_searcher_wrapper(self, temp_store):
        chunk = ChunkMetadata(
            chunk_id="doc2_p1_c0",
            doc_id="doc2",
            filename="comp_sci.pdf",
            page_number=4,
            chunk_index=0,
            text="Reciprocal Rank Fusion merges multiple ranked lists without needing normalized scores.",
            char_count=87,
            token_estimate=12,
        )
        temp_store.index_chunks([chunk])
        searcher = VectorSearcher(temp_store)
        candidates = searcher.search("Tell me about Reciprocal Rank Fusion", top_k=1)
        assert len(candidates) == 1
        assert candidates[0].filename == "comp_sci.pdf"
        assert candidates[0].page_number == 4
        assert candidates[0].to_citation_str() == "[comp_sci.pdf, Page 4]"


class TestAnswerGenerator:
    def test_insufficient_evidence_when_empty_candidates(self):
        generator = AnswerGenerator()
        result = generator.generate_answer("What is the capital of Mars?", candidates=[])
        assert not result.is_sufficient
        assert INSUFFICIENT_EVIDENCE_PHRASE in result.answer
        assert result.citations == []
        assert result.evidence_excerpts == []

    def test_citation_provenance_from_metadata(self):
        generator = AnswerGenerator()
        fake_candidates = [
            RetrievalCandidate(
                chunk_id="docA_p7_c0",
                text="The speed of light in vacuum is approximately 299,792 km per second.",
                doc_id="docA",
                filename="physics.pdf",
                page_number=7,
                chunk_index=0,
                score=0.95,
                rank=1,
                retrieval_mode="vector",
            )
        ]
        result = generator.generate_answer("What is the speed of light?", candidates=fake_candidates)
        # Verify citations were extracted directly from metadata
        assert "[physics.pdf, Page 7]" in result.citations
        assert len(result.evidence_excerpts) == 1
        assert result.evidence_excerpts[0].filename == "physics.pdf"
        assert result.evidence_excerpts[0].page_number == 7
