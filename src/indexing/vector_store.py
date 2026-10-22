"""Persistent ChromaDB vector store wrapper with deduplication and metadata preservation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.indexing.embeddings import BaseEmbedder
from src.ingestion.chunking import ChunkMetadata

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages persistent vector indexing and retrieval via ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "pdf_research_assistant",
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._manifest_path = self.persist_directory / "indexed_documents.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not read manifest, starting fresh: %s", exc)
        return {"_metadata": {}, "documents": {}}

    def _save_manifest(self) -> None:
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save manifest: %s", exc)

    def is_document_indexed(self, doc_id: str) -> bool:
        """Check if doc_id has already been indexed without reprocessing."""
        docs = self._manifest.get("documents", {})
        return doc_id in docs

    def _validate_dimension_compatibility(self) -> None:
        """Ensure current embedder dimension matches existing collection dimensionality."""
        if self.embedder is None or self.collection.count() == 0:
            return

        meta = self._manifest.get("_metadata", {})
        stored_dim = meta.get("dimension")
        stored_model = meta.get("model", "unknown")
        current_dim = self.embedder.get_dimension()
        current_model = getattr(self.embedder, "model_name", "unknown")

        if stored_dim is not None and stored_dim != current_dim:
            raise ValueError(
                f"Embedding dimension mismatch: Chroma collection '{self.collection_name}' contains "
                f"{stored_dim}-dimensional vectors (indexed with model '{stored_model}'), "
                f"but current embedder provides {current_dim}-dimensional vectors (model '{current_model}').\n"
                f"To switch embedding models cleanly, re-index your documents or clear the index."
            )

    def index_chunks(self, chunks: List[ChunkMetadata], force: bool = False) -> int:
        """
        Index chunk list into ChromaDB, skipping if document is already indexed unless forced.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        doc_id = chunks[0].doc_id
        if self.is_document_indexed(doc_id) and not force:
            logger.info("Document %s already indexed. Skipping re-indexing.", doc_id)
            return 0

        if self.embedder is None:
            raise ValueError("Embedder is required to index chunks.")

        # Guard against dimension mismatch
        self._validate_dimension_compatibility()

        # Batch embed documents
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]

        # Chroma upsert guarantees idempotency
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # Update manifest
        if "documents" not in self._manifest:
            self._manifest["documents"] = {}
        if "_metadata" not in self._manifest:
            self._manifest["_metadata"] = {}

        self._manifest["_metadata"] = {
            "dimension": self.embedder.get_dimension(),
            "model": getattr(self.embedder, "model_name", "unknown"),
            "is_fallback": getattr(self.embedder, "is_fallback", False),
        }
        self._manifest["documents"][doc_id] = {
            "filename": chunks[0].filename,
            "chunk_count": len(chunks),
            "indexed_at": str(Path(self.persist_directory).stat().st_mtime),
        }
        self._save_manifest()
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector index for the most similar chunks.

        Returns:
            List of dicts with chunk_id, text, metadata, distance, similarity_score.
        """
        if self.collection.count() == 0:
            return []

        if self.embedder is None:
            raise ValueError("Embedder is required to query vector store.")

        # Guard against dimension mismatch before query
        self._validate_dimension_compatibility()

        query_vec = self.embedder.embed_query(query_text)
        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            dist = float(distances[i]) if distances is not None else 1.0
            # Cosine distance in Chroma is 1 - cosine_similarity (range 0 to 2)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            meta = metadatas[i] if metadatas else {}
            output.append({
                "chunk_id": ids[i],
                "text": docs[i] if docs else "",
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", "unknown"),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "raw_distance": dist,
                "score": similarity,
                "retrieval_mode": "vector",
            })

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Return index summary statistics."""
        docs = self._manifest.get("documents", {})
        meta = self._manifest.get("_metadata", {})
        return {
            "total_chunks": self.collection.count(),
            "indexed_documents": len(docs),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "indexed_dimension": meta.get("dimension"),
            "indexed_model": meta.get("model"),
            "is_fallback": meta.get("is_fallback", False),
        }

    def clear(self) -> None:
        """Clear the collection and manifest."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._manifest = {"_metadata": {}, "documents": {}}
        self._save_manifest()
