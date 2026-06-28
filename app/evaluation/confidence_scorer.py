"""
BankAssist RAG — Confidence Scorer (Phase 9)
==============================================
Computes a multi-dimensional confidence score for every generated answer,
enabling the system to refuse low-confidence responses rather than risk
serving hallucinated or weakly-grounded information to banking customers.

Scoring Dimensions
------------------
1. Retrieval Confidence (weight: 0.35 default)
   - Mean reranker/RRF score of the retrieved chunks.
   - Measures: "How relevant were the documents we found?"
   - Low when: query is out-of-domain, corpus is sparse, or embeddings
     don't capture the concept well.

2. Citation Completeness (weight: 0.35 default)
   - Fraction of [Source N] citations in the answer that resolve to
     actual retrieved chunks.
   - Measures: "Did the model ground its claims in real sources?"
   - Low when: the model invents source numbers, or produces a long
     answer with zero citations.

3. Semantic Grounding / Faithfulness (weight: 0.30 default)
   - For sentences containing numerical claims (rates, amounts, dates),
     computes word overlap between the sentence and its cited source chunk.
   - Measures: "Are the specific facts in the answer actually in the source?"
   - Low when: the model paraphrases numbers incorrectly or fabricates
     policy details.

Aggregation: Weighted Harmonic Mean
------------------------------------
    H = (w1 + w2 + w3) / (w1/s1 + w2/s2 + w3/s3)

Why harmonic mean instead of arithmetic?
  - Arithmetic mean: (0.9 + 0.9 + 0.1) / 3 = 0.63 — looks "okay"
  - Harmonic mean: 3 / (1/0.9 + 1/0.9 + 1/0.1) = 0.25 — correctly flags danger

The harmonic mean is DOMINATED by the lowest score. If any single dimension
is critically low, the overall confidence drops sharply. This is exactly
the behavior we want for a banking system: one bad signal = refuse.

Threshold Enforcement
---------------------
  - overall >= min_threshold (0.40): PASS → answer is served
  - overall < min_threshold:         REFUSE → standard refusal message returned

The threshold is intentionally conservative (0.40, not 0.70) because:
  1. Reranker scores are often in the 0.3–0.7 range for correct answers
  2. BM25/RRF scores have a very different scale
  3. We'd rather serve a slightly uncertain but grounded answer than
     refuse everything — the hallucination guard (Phase 8) catches the
     rest downstream
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config.settings import get_settings
from app.llm.generator import Citation
from app.retriever.pipeline import RetrievalResult
from app.utils.exceptions import ConfidenceBelowThresholdError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConfidenceResult:
    """
    Complete confidence assessment for a generated answer.

    All sub-scores and the aggregated overall score are in [0, 1].
    """

    # Individual dimension scores
    retrieval_confidence: float = 0.0
    citation_completeness: float = 0.0
    faithfulness_score: float = 0.0

    # Aggregated score
    overall_confidence: float = 0.0

    # Threshold and decision
    threshold: float = 0.0
    passed: bool = False

    # Diagnostic details
    details: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence label."""
        if self.overall_confidence >= 0.75:
            return "HIGH"
        elif self.overall_confidence >= 0.50:
            return "MEDIUM"
        elif self.overall_confidence >= 0.35:
            return "LOW"
        else:
            return "VERY_LOW"


# ---------------------------------------------------------------------------
# Confidence Scorer
# ---------------------------------------------------------------------------
class ConfidenceScorer:
    """
    Computes multi-dimensional confidence scores for RAG-generated answers.

    Usage::

        scorer = ConfidenceScorer()
        result = scorer.score(
            answer=generation_result.answer,
            citations=generation_result.citations,
            retrieval_result=retrieval_result,
        )

        if not result.passed:
            # Return refusal instead of answer
            ...
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def score(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
    ) -> ConfidenceResult:
        """
        Compute the full confidence assessment.

        Args:
            answer: The generated answer text.
            citations: Citations extracted from the answer by the generator.
            retrieval_result: The retrieval pipeline output (chunks + scores).

        Returns:
            ConfidenceResult with all sub-scores, overall score, and pass/fail.
        """
        settings = self.settings

        # Dimension 1: Retrieval confidence
        retrieval_conf = self._compute_retrieval_confidence(retrieval_result)

        # Dimension 2: Citation completeness
        citation_comp = self._compute_citation_completeness(
            answer, citations, retrieval_result
        )

        # Dimension 3: Faithfulness / semantic grounding
        faithfulness = self._compute_faithfulness(
            answer, citations, retrieval_result
        )

        # Aggregate via weighted harmonic mean
        overall = self._weighted_harmonic_mean(
            scores=[retrieval_conf, citation_comp, faithfulness],
            weights=[
                settings.confidence_retrieval_weight,
                settings.confidence_citation_weight,
                settings.confidence_generation_weight,
            ],
        )

        threshold = settings.confidence_min_threshold
        passed = overall >= threshold

        result = ConfidenceResult(
            retrieval_confidence=round(retrieval_conf, 4),
            citation_completeness=round(citation_comp, 4),
            faithfulness_score=round(faithfulness, 4),
            overall_confidence=round(overall, 4),
            threshold=threshold,
            passed=passed,
            details={
                "retrieval_chunks": len(retrieval_result.chunks),
                "retrieval_top_score": round(retrieval_result.top_score, 4),
                "retrieval_mean_score": round(retrieval_result.mean_score, 4),
                "citations_found": len(citations),
                "answer_length": len(answer),
                "confidence_label": "",  # Will be set below
            },
        )
        result.details["confidence_label"] = result.confidence_label

        logger.info(
            "confidence_scored",
            retrieval=round(retrieval_conf, 4),
            citation=round(citation_comp, 4),
            faithfulness=round(faithfulness, 4),
            overall=round(overall, 4),
            threshold=threshold,
            passed=passed,
            label=result.confidence_label,
        )

        return result

    def score_and_enforce(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
    ) -> ConfidenceResult:
        """
        Score confidence and raise ConfidenceBelowThresholdError if it fails.

        Use this in the main pipeline to enforce the threshold. The calling
        code should catch the exception and return a refusal message.
        """
        result = self.score(answer, citations, retrieval_result)

        if not result.passed:
            logger.warning(
                "confidence_below_threshold_refusing",
                overall=result.overall_confidence,
                threshold=result.threshold,
            )
            raise ConfidenceBelowThresholdError(
                overall_confidence=result.overall_confidence,
                threshold=result.threshold,
            )

        return result

    # -----------------------------------------------------------------------
    # Dimension 1: Retrieval Confidence
    # -----------------------------------------------------------------------
    def _compute_retrieval_confidence(
        self, retrieval_result: RetrievalResult
    ) -> float:
        """
        Compute retrieval confidence from the retrieved chunk scores.

        Strategy:
          - Use the mean of all chunk scores from the retrieval pipeline.
          - Clamp to [0, 1] since RRF scores can exceed 1.0 in theory.
          - If no chunks, return 0.

        The retrieval scores are already meaningful after reranking (Stage 5).
        Reranker scores via FlagReranker with normalize=True are in [0, 1].
        RRF scores are small positive floats — we normalize them.
        """
        if retrieval_result.is_empty:
            return 0.0

        scores = retrieval_result.scores

        # Determine if scores are from reranker (0–1 range) or RRF (0–0.03 range)
        max_score = max(scores)

        if max_score <= 0:
            return 0.0

        if max_score > 0.5:
            # Reranker scores (sigmoid-normalized, already in [0, 1])
            confidence = sum(scores) / len(scores)
        else:
            # RRF scores — normalize relative to the top score
            # Top-scored chunk gets 1.0, others proportional
            normalized = [s / max_score for s in scores]
            confidence = sum(normalized) / len(normalized)

        return min(max(confidence, 0.0), 1.0)

    # -----------------------------------------------------------------------
    # Dimension 2: Citation Completeness
    # -----------------------------------------------------------------------
    def _compute_citation_completeness(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
    ) -> float:
        """
        Compute what fraction of cited sources are valid and retrievable.

        Components:
          1. Valid citation ratio: cited sources that exist / total cited
          2. Citation presence bonus: penalize answers with no citations
          3. Coverage ratio: fraction of retrieved chunks that were cited

        Final score = weighted combination of above.
        """
        max_valid_source = len(retrieval_result.chunks)

        # Extract all cited source numbers from the answer
        cited_numbers: set[int] = set()
        for match in re.finditer(r'\[Source\s+([\d,\s]+)\]', answer, re.IGNORECASE):
            for num_str in match.group(1).split(","):
                try:
                    cited_numbers.add(int(num_str.strip()))
                except ValueError:
                    pass

        if not cited_numbers:
            # No citations at all
            if len(answer) < 100:
                # Very short answer — might be a direct factual reply, OK
                return 0.6
            else:
                # Long answer without citations — bad
                return 0.2

        # Valid citation ratio
        valid_citations = sum(
            1 for n in cited_numbers if 1 <= n <= max_valid_source
        )
        total_cited = len(cited_numbers)
        valid_ratio = valid_citations / total_cited if total_cited > 0 else 0.0

        # Coverage ratio: what fraction of available chunks were cited
        coverage = min(valid_citations / max(max_valid_source, 1), 1.0)

        # Combine: valid_ratio is more important (0.7) than coverage (0.3)
        score = 0.7 * valid_ratio + 0.3 * coverage

        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------
    # Dimension 3: Faithfulness / Semantic Grounding
    # -----------------------------------------------------------------------
    def _compute_faithfulness(
        self,
        answer: str,
        citations: list[Citation],
        retrieval_result: RetrievalResult,
    ) -> float:
        """
        Check if factual claims in the answer are grounded in cited chunks.

        Focuses on sentences containing numerical values (rates, amounts,
        dates, percentages) — these are the highest-risk claims for
        hallucination in banking.

        For each such sentence:
          - Extract cited source numbers
          - Compute word overlap with the source chunk text
          - If overlap >= min_overlap threshold → grounded
          - Final score = fraction of numerical sentences that are grounded

        If no numerical sentences exist, returns a generous 0.75 (the answer
        is qualitative and harder to fact-check automatically).
        """
        # Build chunk text lookup (1-indexed)
        chunk_texts: dict[int, str] = {}
        for i, chunk in enumerate(retrieval_result.chunks, start=1):
            chunk_texts[i] = chunk.text.lower()

        # Extract sentences with numerical claims
        numerical_sentences = self._extract_numerical_sentences(answer)

        if not numerical_sentences:
            # No numerical claims → can't measure faithfulness objectively
            # Return a moderate score
            return 0.75

        min_overlap = self.settings.hallucination_min_overlap
        grounded_count = 0

        for sentence in numerical_sentences:
            # Find cited sources in this sentence
            cited_in_sentence: set[int] = set()
            for match in re.finditer(
                r'\[Source\s+([\d,\s]+)\]', sentence, re.IGNORECASE
            ):
                for num_str in match.group(1).split(","):
                    try:
                        cited_in_sentence.add(int(num_str.strip()))
                    except ValueError:
                        pass

            if not cited_in_sentence:
                # Numerical claim without citation — assume ungrounded
                continue

            # Check overlap with at least one cited chunk
            sentence_words = set(
                re.findall(r'\b[a-z0-9₹%.,]+\b', sentence.lower())
            )
            if len(sentence_words) < 3:
                grounded_count += 1  # Too short to judge
                continue

            for num in cited_in_sentence:
                if num in chunk_texts:
                    chunk_words = set(
                        re.findall(r'\b[a-z0-9₹%.,]+\b', chunk_texts[num])
                    )
                    overlap = len(sentence_words & chunk_words) / max(
                        len(sentence_words), 1
                    )
                    if overlap >= min_overlap:
                        grounded_count += 1
                        break  # One matching source is sufficient

        score = grounded_count / max(len(numerical_sentences), 1)
        return min(max(score, 0.0), 1.0)

    # -----------------------------------------------------------------------
    # Aggregation: Weighted Harmonic Mean
    # -----------------------------------------------------------------------
    @staticmethod
    def _weighted_harmonic_mean(
        scores: list[float],
        weights: list[float],
    ) -> float:
        """
        Compute weighted harmonic mean of scores.

        H = Σw_i / Σ(w_i / s_i)

        If any score is 0, the harmonic mean would be 0 (division by zero).
        We floor scores at a small epsilon (0.01) to prevent this while
        still producing a very low overall score.
        """
        epsilon = 0.01
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        denominator = 0.0
        for score, weight in zip(scores, weights):
            clamped_score = max(score, epsilon)
            denominator += weight / clamped_score

        if denominator == 0:
            return 0.0

        return total_weight / denominator

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_numerical_sentences(text: str) -> list[str]:
        """Extract sentences that contain numbers, percentages, or currency values."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        numerical_pattern = re.compile(
            r'\d+(?:\.\d+)?'
            r'(?:\s*%|\s*p\.a\.|\s*₹|\s*lakh|\s*crore'
            r'|\s*years|\s*months|\s*days)?'
        )
        return [s for s in sentences if numerical_pattern.search(s)]
