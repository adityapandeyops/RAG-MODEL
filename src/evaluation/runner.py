"""Evaluation runner comparing Vector-only, BM25-only, Hybrid RRF, and Hybrid+Reranker."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    AnswerGenerator,
)
from src.indexing.embeddings import get_embedder
from src.indexing.vector_store import ChromaVectorStore
from src.ingestion.chunking import ChunkMetadata, ParagraphAwareChunker
from src.ingestion.pdf_loader import DocumentMetadata, PageText, PDFLoader
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_search import RetrievalCandidate, VectorSearcher
from src.evaluation.metrics import (
    compute_citation_correctness,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_token_f1,
)

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates comparative RAG evaluation on a fixed dataset."""

    def __init__(
        self,
        questions_path: str = "./data/evaluation_questions.json",
        results_dir: str = "./results",
        chroma_dir: str = "./data/chroma_db_eval",
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.questions_path = Path(questions_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir = Path(chroma_dir)
        self.top_k = top_k
        self.rrf_k = rrf_k

        self.embedder = get_embedder()
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.chroma_dir),
            collection_name="eval_collection",
            embedder=self.embedder,
        )
        self.vector_searcher = VectorSearcher(self.vector_store)
        self.bm25_searcher = BM25Searcher()
        self.fusion = ReciprocalRankFusion(rrf_k=self.rrf_k)
        self.reranker = CrossEncoderReranker(enabled=True)
        self.generator = AnswerGenerator()

    def index_documents(self, documents_info: List[Dict[str, Any]]) -> List[ChunkMetadata]:
        """Index a list of synthetic or real test documents into vector and BM25 stores."""
        chunker = ParagraphAwareChunker(chunk_size=500, chunk_overlap=80)
        loader = PDFLoader()
        all_chunks: List[ChunkMetadata] = []

        self.vector_store.clear()

        for doc_info in documents_info:
            filename = doc_info.get("filename", "eval_doc.pdf")
            content_or_path = doc_info.get("content_or_path")
            pages_list = doc_info.get("pages")

            if isinstance(content_or_path, str) and Path(content_or_path).is_file():
                doc_meta = loader.load_pdf(content_or_path, filename=filename)
            elif pages_list or isinstance(content_or_path, (str, bytes)):
                doc_id = f"eval_{filename.replace('.', '_')}"
                if pages_list is None:
                    pages_list = [str(content_or_path)]
                pages = [
                    PageText(
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_idx + 1,
                        text=page_text,
                        char_count=len(page_text),
                        extraction_status="ok",
                    )
                    for page_idx, page_text in enumerate(pages_list)
                ]
                doc_meta = DocumentMetadata(
                    doc_id=doc_id,
                    filename=filename,
                    total_pages=len(pages),
                    file_hash="eval_hash",
                    file_size_bytes=sum(len(p) for p in pages_list),
                    pages=pages,
                )
            else:
                continue

            chunks = chunker.chunk_document(doc_meta)
            all_chunks.extend(chunks)

        self.vector_store.index_chunks(all_chunks, force=True)
        self.bm25_searcher.fit(all_chunks)
        return all_chunks

    def run_query(self, query: str, config: str) -> Dict[str, Any]:
        """
        Run a single query under a specific configuration:
        'A_vector', 'B_bm25', 'C_hybrid_rrf', 'D_hybrid_reranked'.
        """
        t0 = time.perf_counter()
        retrieval_t0 = time.perf_counter()

        candidates: List[RetrievalCandidate] = []
        rerank_duration = 0.0

        if config == "A_vector":
            candidates = self.vector_searcher.search(query, top_k=self.top_k)
            retrieval_duration = time.perf_counter() - retrieval_t0

        elif config == "B_bm25":
            candidates = self.bm25_searcher.search(query, top_k=self.top_k)
            retrieval_duration = time.perf_counter() - retrieval_t0

        elif config == "C_hybrid_rrf":
            v_cands = self.vector_searcher.search(query, top_k=self.top_k * 2)
            b_cands = self.bm25_searcher.search(query, top_k=self.top_k * 2)
            candidates = self.fusion.fuse(v_cands, b_cands, top_k=self.top_k)
            retrieval_duration = time.perf_counter() - retrieval_t0

        elif config == "D_hybrid_reranked":
            v_cands = self.vector_searcher.search(query, top_k=self.top_k * 2)
            b_cands = self.bm25_searcher.search(query, top_k=self.top_k * 2)
            fused = self.fusion.fuse(v_cands, b_cands, top_k=self.top_k * 2)
            retrieval_duration = time.perf_counter() - retrieval_t0

            rerank_t0 = time.perf_counter()
            candidates, _ = self.reranker.rerank(query, fused, top_k=self.top_k)
            rerank_duration = time.perf_counter() - rerank_t0
        else:
            raise ValueError(f"Unknown config: {config}")

        gen_t0 = time.perf_counter()
        gen_result = self.generator.generate_answer(
            question=query,
            candidates=candidates,
            retrieval_mode=config,
        )
        gen_duration = time.perf_counter() - gen_t0
        total_duration = time.perf_counter() - t0

        return {
            "config": config,
            "query": query,
            "candidates": candidates,
            "generation_result": gen_result,
            "retrieval_latency": retrieval_duration,
            "rerank_latency": rerank_duration,
            "generation_latency": gen_duration,
            "total_latency": total_duration,
        }

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation suite over all questions and all 4 configurations."""
        if not self.questions_path.exists():
            raise FileNotFoundError(f"Evaluation questions file not found: {self.questions_path}")

        with open(self.questions_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        documents_info = eval_data.get("documents", [])
        questions = eval_data.get("questions", [])

        if documents_info:
            self.index_documents(documents_info)

        configs = ["A_vector", "B_bm25", "C_hybrid_rrf", "D_hybrid_reranked"]
        records: List[Dict[str, Any]] = []
        summary: Dict[str, Dict[str, float]] = {c: {} for c in configs}

        for cfg in configs:
            recalls_1, recalls_3, recalls_5, mrrs, cit_scores, f1_scores = [], [], [], [], [], []
            ret_lats, rerank_lats, gen_lats, tot_lats = [], [], [], []

            for q_item in questions:
                qid = q_item["id"]
                query = q_item["question"]
                ref_answer = q_item.get("reference_answer", "")
                rel_chunk_ids: Set[str] = set(q_item.get("relevant_chunk_ids", []))
                expected_pages: Set[int] = set(q_item.get("expected_pages", []))
                expected_files: Set[str] = set(q_item.get("expected_filenames", []))
                is_answerable = q_item.get("answerable", True)

                run_res = self.run_query(query, config=cfg)
                cands = run_res["candidates"]
                gen_res = run_res["generation_result"]

                # Metrics
                r1 = compute_recall_at_k(cands, rel_chunk_ids, k=1) if is_answerable else 1.0
                r3 = compute_recall_at_k(cands, rel_chunk_ids, k=3) if is_answerable else 1.0
                r5 = compute_recall_at_k(cands, rel_chunk_ids, k=5) if is_answerable else 1.0
                mrr = compute_reciprocal_rank(cands, rel_chunk_ids) if is_answerable else 1.0
                cit_score = compute_citation_correctness(gen_res.citations, expected_pages, expected_files)
                f1 = compute_token_f1(gen_res.answer, ref_answer)

                recalls_1.append(r1)
                recalls_3.append(r3)
                recalls_5.append(r5)
                mrrs.append(mrr)
                cit_scores.append(cit_score)
                f1_scores.append(f1)
                ret_lats.append(run_res["retrieval_latency"])
                rerank_lats.append(run_res["rerank_latency"])
                gen_lats.append(run_res["generation_latency"])
                tot_lats.append(run_res["total_latency"])

                records.append({
                    "question_id": qid,
                    "configuration": cfg,
                    "question": query,
                    "answerable": is_answerable,
                    "recall@1": r1,
                    "recall@3": r3,
                    "recall@5": r5,
                    "mrr": mrr,
                    "citation_correctness": cit_score,
                    "token_f1": f1,
                    "retrieval_latency_s": run_res["retrieval_latency"],
                    "rerank_latency_s": run_res["rerank_latency"],
                    "generation_latency_s": run_res["generation_latency"],
                    "total_latency_s": run_res["total_latency"],
                    "retrieved_chunk_ids": [c.chunk_id for c in cands],
                    "citations": gen_res.citations,
                    "generated_answer": gen_res.answer,
                })

            n = max(1, len(questions))
            summary[cfg] = {
                "mean_recall@1": sum(recalls_1) / n,
                "mean_recall@3": sum(recalls_3) / n,
                "mean_recall@5": sum(recalls_5) / n,
                "mean_mrr": sum(mrrs) / n,
                "mean_citation_correctness": sum(cit_scores) / n,
                "mean_token_f1": sum(f1_scores) / n,
                "avg_retrieval_latency_s": sum(ret_lats) / n,
                "avg_rerank_latency_s": sum(rerank_lats) / n,
                "avg_generation_latency_s": sum(gen_lats) / n,
                "avg_total_latency_s": sum(tot_lats) / n,
            }

        output_json_path = self.results_dir / "evaluation_results.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query_results": records}, f, indent=2)

        output_csv_path = self.results_dir / "evaluation_results.csv"
        if records:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec)

        return {"summary": summary, "output_json": str(output_json_path), "output_csv": str(output_csv_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite across all 4 configurations.")
    parser.add_argument("--questions", default="./data/evaluation_questions.json", help="Path to evaluation questions JSON")
    parser.add_argument("--results-dir", default="./results", help="Directory to save evaluation results")
    args = parser.parse_args()

    runner = EvaluationRunner(questions_path=args.questions, results_dir=args.results_dir)
    res = runner.evaluate()
    print("\n--- EVALUATION SUMMARY ---")
    for cfg, metrics in res["summary"].items():
        print(f"\nConfiguration: {cfg}")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")
    print(f"\nResults saved to {res['output_json']} and {res['output_csv']}")
