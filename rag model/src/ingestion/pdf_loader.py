"""PDF Loader module for page-aware text extraction and provenance preservation."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import pypdf


@dataclass
class PageText:
    """Represents extracted text from a single PDF page with provenance."""
    doc_id: str
    filename: str
    page_number: int  # 1-based page number for UI and citations
    text: str
    char_count: int
    extraction_status: str  # "ok", "empty_page", "low_text"
    warning: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Document level metadata."""
    doc_id: str
    filename: str
    total_pages: int
    file_hash: str
    file_size_bytes: int
    pages: List[PageText] = field(default_factory=list)
    extraction_error: Optional[str] = None


class PDFLoader:
    """Extracts text page-by-page from PDFs while strictly preserving source provenance."""

    def __init__(self, low_text_threshold: int = 50):
        self.low_text_threshold = low_text_threshold

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute stable SHA256 hex digest for document identity and deduplication."""
        return hashlib.sha256(content).hexdigest()

    def load_pdf(
        self,
        source: Union[str, Path, BinaryIO, bytes],
        filename: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Load and extract text page-by-page from a PDF file or stream.

        Args:
            source: Path to file, file-like binary buffer, or raw bytes.
            filename: Optional explicit filename (inferred from path if not given).

        Returns:
            DocumentMetadata with all page-level text and provenance.
        """
        # Read bytes and determine filename
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                return DocumentMetadata(
                    doc_id="error_missing_file",
                    filename=str(path.name),
                    total_pages=0,
                    file_hash="",
                    file_size_bytes=0,
                    extraction_error=f"File not found: {path}",
                )
            resolved_filename = filename or path.name
            with open(path, "rb") as f:
                content = f.read()
        elif isinstance(source, bytes):
            resolved_filename = filename or "uploaded_document.pdf"
            content = source
        elif hasattr(source, "read"):
            resolved_filename = filename or getattr(source, "name", "uploaded_document.pdf")
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            return DocumentMetadata(
                doc_id="error_invalid_input",
                filename=filename or "unknown.pdf",
                total_pages=0,
                file_hash="",
                file_size_bytes=0,
                extraction_error=f"Unsupported source type: {type(source)}",
            )

        file_size = len(content)
        if file_size == 0:
            return DocumentMetadata(
                doc_id="empty_file",
                filename=resolved_filename,
                total_pages=0,
                file_hash="",
                file_size_bytes=0,
                extraction_error="File is empty (0 bytes).",
            )

        file_hash = self.compute_sha256(content)
        doc_id = f"{file_hash[:12]}_{resolved_filename.replace(' ', '_')}"

        stream = io.BytesIO(content)
        try:
            reader = pypdf.PdfReader(stream)
        except Exception as exc:
            return DocumentMetadata(
                doc_id=doc_id,
                filename=resolved_filename,
                total_pages=0,
                file_hash=file_hash,
                file_size_bytes=file_size,
                extraction_error=f"Failed to parse PDF: {str(exc)}",
            )

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    return DocumentMetadata(
                        doc_id=doc_id,
                        filename=resolved_filename,
                        total_pages=0,
                        file_hash=file_hash,
                        file_size_bytes=file_size,
                        extraction_error="PDF is encrypted and password is required.",
                    )
            except Exception as exc:
                return DocumentMetadata(
                    doc_id=doc_id,
                    filename=resolved_filename,
                    total_pages=0,
                    file_hash=file_hash,
                    file_size_bytes=file_size,
                    extraction_error=f"Encrypted PDF cannot be read: {str(exc)}",
                )

        total_pages = len(reader.pages)
        pages: List[PageText] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1  # 1-based page index for UI and citations
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        filename=resolved_filename,
                        page_number=page_num,
                        text="",
                        char_count=0,
                        extraction_status="error",
                        warning=f"Page extraction error: {str(exc)}",
                    )
                )
                continue

            clean_text = page_text.strip()
            char_count = len(clean_text)

            if char_count == 0:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        filename=resolved_filename,
                        page_number=page_num,
                        text="",
                        char_count=0,
                        extraction_status="empty_page",
                        warning="Page contains no extractable text (blank or scanned).",
                    )
                )
            elif char_count < self.low_text_threshold:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        filename=resolved_filename,
                        page_number=page_num,
                        text=clean_text,
                        char_count=char_count,
                        extraction_status="low_text",
                        warning=f"Page has very low text ({char_count} chars). May be an image/diagram.",
                    )
                )
            else:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        filename=resolved_filename,
                        page_number=page_num,
                        text=clean_text,
                        char_count=char_count,
                        extraction_status="ok",
                    )
                )

        return DocumentMetadata(
            doc_id=doc_id,
            filename=resolved_filename,
            total_pages=total_pages,
            file_hash=file_hash,
            file_size_bytes=file_size,
            pages=pages,
        )
