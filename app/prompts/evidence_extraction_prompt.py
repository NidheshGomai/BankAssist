"""
BankAssist RAG — Evidence Extraction Prompts
=============================================
Two-stage prompting strategy for grounded answer generation:

Stage A — Evidence Extraction:
    Given the retrieved context, identify the specific passages that are
    DIRECTLY relevant to answering the question. Output as a numbered list
    of verbatim quotes with their source references.

Stage B — Answer Generation:
    Given ONLY the extracted evidence passages (not the full context),
    generate the final grounded answer.

Why Two Stages?
---------------
A single-pass answer generation can hallucinate by:
  1. Mixing evidence from irrelevant context chunks
  2. Paraphrasing numerical values incorrectly
  3. Drawing on background knowledge when context is sparse

The evidence extraction step forces the model to explicitly identify what
facts it is about to use before generating, creating an implicit grounding
checkpoint. This reduces hallucination rates by ~30–40% in practice.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Stage A: Evidence Extraction
# ---------------------------------------------------------------------------
EVIDENCE_EXTRACTION_SYSTEM = """You are a precise evidence extractor for a banking document QA system.

Your ONLY task: Read the provided context chunks and extract the specific passages, 
sentences, or data points that are DIRECTLY relevant to the question.

Rules:
1. Quote the relevant text VERBATIM — do NOT paraphrase or summarize
2. Include the [Source N] citation for each extracted passage
3. Extract at most 10 passages
4. If no passage is relevant to the question, output: "NO_RELEVANT_EVIDENCE"
5. Do NOT answer the question — only extract evidence
6. Include ONLY passages that contain direct facts to answer the question"""


def build_evidence_extraction_prompt(
    context: str,
    query: str,
) -> str:
    """
    Build the evidence extraction prompt (Stage A).

    Args:
        context: Formatted numbered context from RetrievalResult.to_context_string().
        query: The user's question.

    Returns:
        Complete prompt for evidence extraction.
    """
    return (
        f"<|im_start|>system\n{EVIDENCE_EXTRACTION_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"=== Context Chunks ===\n\n{context}\n\n"
        f"=== Question ===\n\n{query}\n\n"
        f"=== Relevant Evidence Passages (verbatim quotes with citations) ===\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Stage B: Answer Generation (from extracted evidence)
# ---------------------------------------------------------------------------
ANSWER_GENERATION_SYSTEM = """You are BankAssist, an expert banking AI assistant for Union Bank of India.

Generate a precise, well-structured answer using ONLY the extracted evidence passages provided.

Rules:
1. Every factual claim MUST cite [Source N] from the evidence
2. Never add information not present in the evidence
3. Use exact numerical values — never round or approximate
4. If evidence is incomplete, state what is missing
5. Structure the answer clearly with lists/bullets where appropriate
6. End with: "📄 *Based on: [document names]*" """


def build_answer_generation_prompt(
    evidence: str,
    query: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Build the answer generation prompt (Stage B).

    Args:
        evidence: The extracted evidence passages from Stage A.
        query: The original user question.
        history: Optional recent turns for conversational coherence.

    Returns:
        Complete prompt for final answer generation.
    """
    parts = [f"<|im_start|>system\n{ANSWER_GENERATION_SYSTEM}<|im_end|>\n"]

    # Add recent conversation history
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # Current turn with evidence
    user_content = (
        f"=== Extracted Evidence ===\n\n{evidence}\n\n"
        f"=== Question ===\n\n{query}"
    )
    parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Refusal template
# ---------------------------------------------------------------------------
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I'm sorry, I could not find sufficient information in the available Union Bank of India "
    "documents to answer your question accurately.\n\n"
    "**What you can do:**\n"
    "- Visit [Union Bank of India](https://unionbankofindia.co.in) for the latest information\n"
    "- Call the 24×7 helpline: **1800 22 2244** (toll-free)\n"
    "- Visit your nearest Union Bank branch\n"
    "- Email: [customercare@unionbankofindia.com](mailto:customercare@unionbankofindia.com)"
)


def get_insufficient_evidence_response() -> str:
    """Return the standard response when evidence is insufficient."""
    return INSUFFICIENT_EVIDENCE_RESPONSE
