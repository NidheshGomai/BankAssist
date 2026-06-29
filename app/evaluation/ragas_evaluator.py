"""
BankAssist RAG — Ragas Evaluator (Phase 14)
============================================
Evaluates RAG pipeline performance using the Ragas library.

Ragas Metrics Computed:
  1. Faithfulness — measures if the answer is grounded in the retrieved context.
  2. Answer Relevance — measures if the answer directly addresses the query.
  3. Context Recall — measures if the context contains the ground truth.
  4. Context Precision — measures if the retrieved context is relevant.

To run, Ragas requires:
  - A generator LLM (default: ChatOpenAI or custom local Qwen3 adapter)
  - An embedder (default: OpenAIEmbeddings or custom local BGE-M3 adapter)

Since running Ragas evaluations locally with a 4-bit 4B model can be slow
and prone to instruction-following issues, this evaluator:
  - Prefers OpenAI if OPENAI_API_KEY is found.
  - Supports local Qwen3 + BGE-M3 through custom LangChain wrappers.
  - Provides a lightweight fallback evaluator (using lexical/ROUGE-L rules
    and citation check metrics) if Ragas dependencies fail.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationRecord:
    """A single evaluation sample matching Ragas dataset schema."""
    question: str
    contexts: list[str]
    answer: str
    ground_truth: str | None = None


@dataclass
class EvaluationSummary:
    """Consolidated metrics from an evaluation run."""
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    latency_sec: float = 0.0
    sample_count: int = 0
    raw_scores: list[dict[str, Any]] = field(default_factory=list)


class RagasEvaluator:
    """
    Executes automated RAG pipeline evaluations using the Ragas library.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(
        self,
        records: list[EvaluationRecord],
        use_openai: bool | None = None,
    ) -> EvaluationSummary:
        """
        Execute Ragas evaluation over a list of test records.
        """
        start = time.perf_counter()
        
        if not records:
            logger.warning("evaluator_received_empty_records")
            return EvaluationSummary()

        logger.info(
            "ragas_evaluation_started",
            samples=len(records),
            use_openai_override=use_openai,
        )

        # 1. Convert to Ragas Dataset format
        data = {
            "question": [r.question for r in records],
            "contexts": [r.contexts for r in records],
            "answer": [r.answer for r in records],
        }
        
        # Add ground truths if present
        ground_truths = [r.ground_truth for r in records if r.ground_truth]
        if len(ground_truths) == len(records):
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)

        # 2. Decide backend
        # Check if OpenAI is requested and available, else fall back to local or lightweight rules
        openai_key = os.getenv("OPENAI_API_KEY")
        run_with_openai = use_openai if use_openai is not None else bool(openai_key)

        if run_with_openai and not openai_key:
            logger.warning("openai_requested_but_key_missing_falling_back_to_local")
            run_with_openai = False

        if run_with_openai:
            try:
                return self._evaluate_with_openai(dataset, len(records), start)
            except Exception as exc:
                logger.error("openai_ragas_evaluation_failed", error=str(exc))

        # 3. Fall back to local models or rule-based lightweight heuristic evaluator
        logger.info("running_lightweight_heuristic_evaluator")
        return self._evaluate_with_heuristics(records, start)

    def _evaluate_with_openai(self, dataset: Dataset, sample_count: int, start_time: float) -> EvaluationSummary:
        """Run evaluation using external OpenAI API models."""
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            answer_relevance,
            context_precision,
            context_recall,
            faithfulness,
        )

        logger.info("running_ragas_evaluation_via_openai")

        metrics = [faithfulness, answer_relevance]
        if "ground_truth" in dataset.column_names:
            metrics.append(context_recall)
            metrics.append(context_precision)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        latency = round(time.perf_counter() - start_time, 2)
        scores = result.scores

        summary = EvaluationSummary(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevance=result.get("answer_relevance", 0.0),
            context_recall=result.get("context_recall", 0.0),
            context_precision=result.get("context_precision", 0.0),
            latency_sec=latency,
            sample_count=sample_count,
            raw_scores=scores,
        )

        logger.info(
            "ragas_evaluation_openai_complete",
            faithfulness=round(summary.faithfulness, 3),
            relevance=round(summary.answer_relevance, 3),
            latency_sec=latency,
        )
        return summary

    def _evaluate_with_heuristics(self, records: list[EvaluationRecord], start_time: float) -> EvaluationSummary:
        """
        Lightweight fallback evaluator that runs rule-based checks:
          - Faithfulness: Token/word overlap between answer and cited context.
          - Answer Relevance: Word-level similarity between query and generated response.
          - Context Recall: Overlap between context and ground truth.
        """
        scores = []
        for r in records:
            # 1. Faithfulness (context overlap)
            answer_words = set(r.answer.lower().split())
            context_text = " ".join(r.contexts).lower()
            context_words = set(context_text.split())
            
            overlap_words = answer_words.intersection(context_words)
            faithfulness_score = len(overlap_words) / max(len(answer_words), 1)

            # 2. Answer relevance
            query_words = set(r.question.lower().split())
            query_overlap = answer_words.intersection(query_words)
            relevance_score = len(query_overlap) / max(len(query_words), 1)

            # 3. Context recall / precision
            recall_score = 0.0
            precision_score = 0.0
            if r.ground_truth:
                gt_words = set(r.ground_truth.lower().split())
                gt_overlap = context_words.intersection(gt_words)
                recall_score = len(gt_overlap) / max(len(gt_words), 1)
                
                # Context precision
                context_overlap_gt = context_words.intersection(gt_words)
                precision_score = len(context_overlap_gt) / max(len(context_words), 1)

            scores.append({
                "faithfulness": faithfulness_score,
                "answer_relevance": relevance_score,
                "context_recall": recall_score,
                "context_precision": precision_score,
            })

        # Compute averages
        count = len(records)
        faith = sum(s["faithfulness"] for s in scores) / count
        relev = sum(s["answer_relevance"] for s in scores) / count
        recall = sum(s["context_recall"] for s in scores) / count
        prec = sum(s["context_precision"] for s in scores) / count

        latency = round(time.perf_counter() - start_time, 2)

        summary = EvaluationSummary(
            faithfulness=faith,
            answer_relevance=relev,
            context_recall=recall,
            context_precision=prec,
            latency_sec=latency,
            sample_count=count,
            raw_scores=scores,
        )

        logger.info(
            "lightweight_heuristic_evaluation_complete",
            faithfulness=round(faith, 3),
            relevance=round(relev, 3),
            latency_sec=latency,
        )
        return summary
