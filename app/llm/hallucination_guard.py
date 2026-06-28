"""
BankAssist RAG — Hallucination Guard
=======================================
Post-generation verification layer that validates generated answers
against the retrieved context before they are returned to the user.

Three Checks
------------
1. Citation Presence Check:
   Every [Source N] cited in the answer must correspond to an actual
   retrieved chunk. Invented source numbers are flagged.

2. Faithfulness Check (Answer–Context Overlap):
   Extracts factual sentences from the answer and verifies that each
   has sufficient lexical overlap with the cited source chunk.
   This catches cases where the LLM slightly paraphrases a number
   differently from what the source says.

3. Confidence Threshold Enforcement:
   If overall confidence score < min_threshold, the answer is suppressed
   and an InsufficientEvidence refusal is returned instead.

Severity Levels
---------------
- PASS: Answer is clean, return as-is
- WARN: Minor issues detected, answer returned with a disclaimer appended
- FAIL: Hallucination detected, answer suppressed

Design Philosophy
-----------------
This is a heuristic guard, not a perfect detector. It is calibrated to
minimize false positives (blocking correct answers) while catching the
most common hallucination patterns in banking RAG systems:
  - Invented policy clauses
  - Wrong interest rates
  - Fabricated eligibility criteria
"""

from __future__ import annotations

import re
from enum import Enum

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.llm.generator import Citation
from app.retriever.pipeline import RetrievalResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GuardResult(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


_WARNING_DISCLAIMER = (
    "\n\n---\n⚠️ *Note: This response may contain information that could not be fully "
    "verified against the source documents. Please cross-check with official bank documentation "
    "or contact Union Bank of India directly before making any financial decisions.*"
)


class HallucinationGuard:
    """
    Post-generation verification guard.

    Usage::

        guard = HallucinationGuard()
        status, verified_answer = guard.verify(
            answer=generation_result.answer,
            citations=generation_result.citations,
            retrieval_result=retrieval_result,
            confidence=overall_confidence,
        )
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def verify(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
        confidence: float,
    ) -> tuple[GuardResult, str]:
        """
        Run all hallucination checks on the generated answer.

        Returns:
            (GuardResult, verified_answer_string)
            If FAIL, verified_answer is the refusal message.
            If WARN, verified_answer has a disclaimer appended.
            If PASS, verified_answer is the original answer unchanged.
        """
        if not self.settings.hallucination_guard_enabled:
            return GuardResult.PASS, answer

        issues: list[str] = []

        # Check 1: Citation presence
        citation_ok, citation_issues = self._check_citations(answer, retrieval_result)
        issues.extend(citation_issues)

        # Check 2: Faithfulness (sentence–chunk overlap)
        faithfulness_ok, faithfulness_issues = self._check_faithfulness(
            answer, citations, retrieval_result
        )
        issues.extend(faithfulness_issues)

        # Check 3: Confidence threshold
        if confidence < self.settings.confidence_min_threshold:
            issues.append(
                f"Confidence {confidence:.3f} below threshold {self.settings.confidence_min_threshold}"
            )

        # Determine severity
        if not issues:
            logger.debug("hallucination_guard_pass")
            return GuardResult.PASS, answer

        # Log all issues
        logger.warning("hallucination_guard_issues_detected", issues=issues)

        # Fatal: multiple issues or extremely low confidence → FAIL
        fatal_confidence = confidence < (self.settings.confidence_min_threshold * 0.7)
        many_issues = len(issues) >= 3

        if fatal_confidence or many_issues:
            logger.error(
                "hallucination_guard_fail",
                confidence=confidence,
                num_issues=len(issues),
            )
            from app.prompts.evidence_extraction_prompt import (  # noqa: PLC0415
                get_insufficient_evidence_response,
            )
            return GuardResult.FAIL, get_insufficient_evidence_response()

        # Warn: minor issues → append disclaimer
        logger.warning("hallucination_guard_warn", num_issues=len(issues))
        return GuardResult.WARN, answer + _WARNING_DISCLAIMER

    # -----------------------------------------------------------------------
    # Check 1: Citation presence
    # -----------------------------------------------------------------------
    def _check_citations(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
    ) -> tuple[bool, list[str]]:
        """Verify all cited source numbers exist in the retrieval result."""
        issues = []
        max_valid = len(retrieval_result.chunks)

        cited_numbers = set()
        for match in re.finditer(r'\[Source\s+([\d,\s]+)\]', answer, re.IGNORECASE):
            for num_str in match.group(1).split(","):
                try:
                    cited_numbers.add(int(num_str.strip()))
                except ValueError:
                    pass

        for num in cited_numbers:
            if num < 1 or num > max_valid:
                issues.append(f"Invalid citation [Source {num}] — only {max_valid} sources available")

        if self.settings.require_citations and not cited_numbers and len(answer) > 100:
            issues.append("Answer contains no citations despite substantive content")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------------
    # Check 2: Faithfulness (sentence overlap)
    # -----------------------------------------------------------------------
    def _check_faithfulness(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
    ) -> tuple[bool, list[str]]:
        """
        For each sentence in the answer that contains a number/percentage/rate,
        verify it has sufficient word overlap with the cited source chunk.
        """
        if not citations:
            return True, []

        # Build chunk lookup: source_number → chunk_text
        chunk_texts: dict[int, str] = {}
        for i, chunk in enumerate(retrieval_result.chunks, start=1):
            chunk_texts[i] = chunk.text.lower()

        issues = []
        min_overlap = self.settings.hallucination_min_overlap

        # Extract sentences containing numerical claims
        numerical_sentences = self._extract_numerical_sentences(answer)

        for sentence in numerical_sentences:
            cited_in_sentence = set()
            for match in re.finditer(r'\[Source\s+([\d,\s]+)\]', sentence, re.IGNORECASE):
                for num_str in match.group(1).split(","):
                    try:
                        cited_in_sentence.add(int(num_str.strip()))
                    except ValueError:
                        pass

            if not cited_in_sentence:
                continue  # Uncited numerical claim — handled by citation check

            # Check overlap with at least one cited chunk
            sentence_words = set(
                re.findall(r'\b[a-z0-9₹%.,]+\b', sentence.lower())
            )
            if len(sentence_words) < 3:
                continue

            found_overlap = False
            for num in cited_in_sentence:
                if num in chunk_texts:
                    chunk_words = set(
                        re.findall(r'\b[a-z0-9₹%.,]+\b', chunk_texts[num])
                    )
                    overlap = len(sentence_words & chunk_words) / max(len(sentence_words), 1)
                    if overlap >= min_overlap:
                        found_overlap = True
                        break

            if not found_overlap:
                issues.append(
                    f"Low faithfulness: sentence not grounded in cited source — '{sentence[:80]}...'"
                )

        return len(issues) == 0, issues

    @staticmethod
    def _extract_numerical_sentences(text: str) -> list[str]:
        """Extract sentences that contain numbers, percentages, or currency values."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        numerical_pattern = re.compile(
            r'\d+(?:\.\d+)?(?:\s*%|\s*p\.a\.|\s*₹|\s*lakh|\s*crore|\s*years|\s*months|\s*days)?'
        )
        return [s for s in sentences if numerical_pattern.search(s)]
