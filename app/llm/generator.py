"""
BankAssist RAG — Answer Generator
====================================
Orchestrates the two-stage (evidence extraction → answer generation) LLM pipeline
and provides both synchronous and streaming interfaces.

Pipeline
--------
  Input: query + RetrievalResult (chunks + scores)
         ↓
  Stage A: Evidence Extraction
    - Extracts verbatim evidence passages from context chunks
    - Returns: evidence_text (quoted passages with citations)
         ↓
  Stage B: Answer Generation
    - Generates final answer grounded on extracted evidence
    - Includes [Source N] citations
    - Returns: GenerationResult (answer, citations, confidence metadata)

Streaming
---------
For the FastAPI /chat endpoint, `stream_answer()` yields SSE-formatted
JSON strings as tokens arrive. The frontend can render tokens in real-time
while citations are attached at the end.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch

from app.config.settings import get_settings
from app.llm.qwen3_loader import get_qwen3_model
from app.prompts.evidence_extraction_prompt import (
    build_answer_generation_prompt,
    build_evidence_extraction_prompt,
    get_insufficient_evidence_response,
)
from app.retriever.pipeline import RetrievalResult
from app.utils.exceptions import LLMInferenceError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """A single citation extracted from the generated answer."""
    source_number: int
    doc_title: str
    section_path: str
    page_number: int
    doc_category: str
    source_url: str
    chunk_id: str


@dataclass
class GenerationResult:
    """Complete output of the answer generator."""
    answer: str = ""
    evidence: str = ""
    citations: list[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    evidence_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    input_chunks: int = 0
    was_refused: bool = False
    refusal_reason: str = ""


class AnswerGenerator:
    """
    Two-stage answer generator grounded on retrieved context chunks.

    Usage (synchronous)::

        generator = AnswerGenerator()
        result = generator.generate(query, retrieval_result, history)

    Usage (streaming)::

        for sse_chunk in generator.stream_answer(query, retrieval_result, history):
            yield sse_chunk
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = get_qwen3_model()

    # -----------------------------------------------------------------------
    # Synchronous generation
    # -----------------------------------------------------------------------
    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: list[dict[str, str]] | None = None,
    ) -> GenerationResult:
        """
        Generate a grounded answer using two-stage prompting.

        Returns:
            GenerationResult with answer, evidence, citations, and timing.
        """
        start = time.perf_counter()
        history = history or []

        if retrieval_result.is_empty:
            return GenerationResult(
                answer=get_insufficient_evidence_response(),
                was_refused=True,
                refusal_reason="no_chunks_retrieved",
            )

        context = retrieval_result.to_context_string(max_chunks=self.settings.llm_max_context_chunks)

        # Stage A: Evidence extraction
        t0 = time.perf_counter()
        evidence = self._extract_evidence(context, query)
        evidence_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not evidence or evidence.strip().upper() == "NO_RELEVANT_EVIDENCE":
            logger.warning("evidence_extraction_found_none", query=query[:80])
            return GenerationResult(
                answer=get_insufficient_evidence_response(),
                was_refused=True,
                refusal_reason="no_relevant_evidence",
                evidence_latency_ms=evidence_ms,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )

        # Stage B: Answer generation
        t0 = time.perf_counter()
        # Free VRAM between LLM calls (critical for 6GB GPUs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        answer = self._generate_answer(evidence, query, history)
        answer_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Extract citations from answer
        citations = self._extract_citations(answer, retrieval_result)

        total_ms = round((time.perf_counter() - start) * 1000, 1)

        logger.info(
            "answer_generated",
            query=query[:80],
            answer_len=len(answer),
            citations_found=len(citations),
            evidence_ms=evidence_ms,
            answer_ms=answer_ms,
            total_ms=total_ms,
        )

        return GenerationResult(
            answer=answer,
            evidence=evidence,
            citations=citations,
            latency_ms=total_ms,
            evidence_latency_ms=evidence_ms,
            answer_latency_ms=answer_ms,
            input_chunks=len(retrieval_result.chunks),
        )

    # -----------------------------------------------------------------------
    # Streaming answer (for SSE)
    # -----------------------------------------------------------------------
    def stream_answer(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: list[dict[str, str]] | None = None,
    ):
        """
        Stream the answer token-by-token as JSON-formatted SSE strings.

        Yields:
            JSON strings: {"type": "token", "content": "..."} during streaming.
            Final:        {"type": "done", "citations": [...], "confidence": float}
        """
        import json  # noqa: PLC0415
        history = history or []

        if retrieval_result.is_empty:
            yield json.dumps({
                "type": "error",
                "content": get_insufficient_evidence_response(),
                "refused": True,
            })
            return

        context = retrieval_result.to_context_string(max_chunks=self.settings.llm_max_context_chunks)

        # Stage A: Evidence extraction (non-streaming, fast)
        evidence = self._extract_evidence(context, query)

        if not evidence or evidence.strip().upper() == "NO_RELEVANT_EVIDENCE":
            yield json.dumps({
                "type": "error",
                "content": get_insufficient_evidence_response(),
                "refused": True,
            })
            return

        # Stage B: Streaming answer generation
        answer_prompt = build_answer_generation_prompt(evidence, query, history)
        full_answer = []

        try:
            for token in self._llm.stream_generate(answer_prompt):
                full_answer.append(token)
                yield json.dumps({"type": "token", "content": token})

            complete_answer = "".join(full_answer)
            citations = self._extract_citations(complete_answer, retrieval_result)

            yield json.dumps({
                "type": "done",
                "citations": [
                    {
                        "source_number": c.source_number,
                        "doc_title": c.doc_title,
                        "section": c.section_path,
                        "page": c.page_number,
                        "url": c.source_url,
                    }
                    for c in citations
                ],
                "context_score": round(retrieval_result.mean_score, 4),
            })

        except Exception as exc:
            logger.error("stream_answer_failed", error=str(exc))
            yield json.dumps({"type": "error", "content": "Generation failed. Please try again."})

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _extract_evidence(self, context: str, query: str) -> str:
        """Run Stage A: evidence extraction."""
        try:
            prompt = build_evidence_extraction_prompt(context, query)
            evidence = self._llm.generate_text(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.05,
                do_sample=False,
            )
            logger.debug("evidence_extracted", evidence_len=len(evidence))
            return evidence.strip()
        except Exception as exc:
            logger.warning("evidence_extraction_failed", error=str(exc))
            return ""  # Fallback to direct answer generation

    def _generate_answer(
        self,
        evidence: str,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """Run Stage B: answer generation from extracted evidence."""
        try:
            prompt = build_answer_generation_prompt(evidence, query, history)
            answer = self._llm.generate_text(
                prompt=prompt,
                max_new_tokens=self.settings.llm_max_new_tokens,
                temperature=self.settings.llm_temperature,
                do_sample=self.settings.llm_do_sample,
            )
            return answer.strip()
        except Exception as exc:
            logger.error("answer_generation_failed", error=str(exc))
            raise LLMInferenceError(f"Answer generation failed: {exc}") from exc

    def _extract_citations(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
    ) -> list[Citation]:
        """
        Extract [Source N] citations from the answer and resolve them to chunks.
        """
        # Find all [Source N] or [Source N, M] patterns
        pattern = re.compile(r'\[Source\s+([\d,\s]+)\]', re.IGNORECASE)
        cited_numbers: set[int] = set()

        for match in pattern.finditer(answer):
            for num_str in match.group(1).split(","):
                try:
                    cited_numbers.add(int(num_str.strip()))
                except ValueError:
                    pass

        # Map source numbers (1-indexed) to chunks
        citations = []
        for num in sorted(cited_numbers):
            idx = num - 1  # Convert to 0-indexed
            if 0 <= idx < len(retrieval_result.chunks):
                chunk = retrieval_result.chunks[idx]
                citations.append(Citation(
                    source_number=num,
                    doc_title=chunk.doc_title,
                    section_path=chunk.section_path,
                    page_number=chunk.page_number,
                    doc_category=chunk.doc_category,
                    source_url=chunk.source_url,
                    chunk_id=chunk.chunk_id,
                ))

        return citations
