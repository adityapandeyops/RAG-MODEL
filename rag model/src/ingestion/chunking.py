"""Chunking module supporting paragraph-aware text splitting with metadata preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.ingestion.pdf_loader import DocumentMetadata, PageText


@dataclass
class ChunkMetadata:
    """Represents a text chunk with complete provenance metadata."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int  # 1-based page number
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
        }


class ParagraphAwareChunker:
    """Splits document text into paragraph-aware chunks while preserving page provenance."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_size <= 50:
            raise ValueError(f"chunk_size must be > 50, got {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_units(self, text: str) -> List[str]:
        """Split text by paragraphs, then sentences if paragraphs are large."""
        paragraphs = re.split(r"\n\s*\n+", text.strip())
        units: List[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                # Split large paragraph by sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_sent_group = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current_sent_group) + len(s) + 1 <= self.chunk_size:
                        current_sent_group = f"{current_sent_group} {s}".strip() if current_sent_group else s
                    else:
                        if current_sent_group:
                            units.append(current_sent_group)
                        if len(s) > self.chunk_size:
                            # Hard split words if sentence is excessively long
                            words = s.split()
                            word_chunk = ""
                            for w in words:
                                if len(word_chunk) + len(w) + 1 <= self.chunk_size:
                                    word_chunk = f"{word_chunk} {w}".strip() if word_chunk else w
                                else:
                                    if word_chunk:
                                        units.append(word_chunk)
                                    word_chunk = w
                            if word_chunk:
                                units.append(word_chunk)
                            current_sent_group = ""
                        else:
                            current_sent_group = s
                if current_sent_group:
                    units.append(current_sent_group)

        return units

    def chunk_page(self, page: PageText, starting_chunk_index: int = 0) -> List[ChunkMetadata]:
        """Chunk a single page and return chunks tagged with page metadata."""
        if not page.text or page.extraction_status == "empty_page":
            return []

        units = self._split_into_units(page.text)
        if not units:
            return []

        chunks: List[ChunkMetadata] = []
        current_text = ""
        chunk_idx = starting_chunk_index

        for unit in units:
            if not current_text:
                current_text = unit
            elif len(current_text) + len(unit) + 2 <= self.chunk_size:
                current_text = f"{current_text}\n\n{unit}"
            else:
                # Finalize current chunk
                token_est = max(1, int(len(current_text.split()) * 1.3))
                chunk_id = f"{page.doc_id}_p{page.page_number}_c{chunk_idx}"
                chunks.append(
                    ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=page.doc_id,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        text=current_text,
                        char_count=len(current_text),
                        token_estimate=token_est,
                    )
                )
                chunk_idx += 1

                # Carry over overlap if possible
                overlap_text = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else current_text
                # Try to keep boundary at whitespace
                space_pos = overlap_text.find(" ")
                if space_pos != -1 and space_pos < len(overlap_text) - 1:
                    overlap_text = overlap_text[space_pos + 1:]
                current_text = f"{overlap_text}\n\n{unit}".strip()

        if current_text:
            token_est = max(1, int(len(current_text.split()) * 1.3))
            chunk_id = f"{page.doc_id}_p{page.page_number}_c{chunk_idx}"
            chunks.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    text=current_text,
                    char_count=len(current_text),
                    token_estimate=token_est,
                )
            )

        return chunks

    def chunk_document(self, doc: DocumentMetadata) -> List[ChunkMetadata]:
        """Chunk all pages of a document, maintaining sequential chunk indexing."""
        all_chunks: List[ChunkMetadata] = []
        chunk_counter = 0

        for page in doc.pages:
            page_chunks = self.chunk_page(page, starting_chunk_index=chunk_counter)
            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        return all_chunks
