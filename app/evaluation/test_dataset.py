"""
BankAssist RAG — Evaluation Dataset (Phase 14)
================================================
Defines the golden test dataset of questions, category tags, and expected
ground truth answers used to validate the RAG pipeline's accuracy,
correctness, and lack of hallucination.

The dataset includes questions covering all banking domains:
  - Retail products (savings, loans, cards)
  - Corporate products (MSME schemes, guarantees)
  - NRI-specific regulations
  - Grievance escalation workflows
  - Interest rate inquiries
"""

from __future__ import annotations

from app.evaluation.ragas_evaluator import EvaluationRecord

# ---------------------------------------------------------------------------
# Golden test dataset (Questions + Ground Truth Answers)
# ---------------------------------------------------------------------------
GOLDEN_DATASET: list[dict[str, str]] = [
    {
        "question": "What is the minimum balance required for a Union Bank Savings Account?",
        "ground_truth": "The minimum balance requirements for a Union Bank Savings Account depend on the branch location: ₹1,000 for Metro branches, ₹500 for Urban branches, and ₹250 for Rural/Semi-Urban branches.",
        "category": "retail",
    },
    {
        "question": "What documents are required to apply for a Union Bank Home Loan?",
        "ground_truth": "Applying for a home loan requires: completed application form, passport-size photographs, identity proof (Aadhaar/PAN/Passport), address proof, latest 3 months salary slips, Form 16, and property-related documents (sale deed, NOC, approved plan).",
        "category": "retail",
    },
    {
        "question": "How can a customer escalate a complaint if the branch manager doesn't resolve it?",
        "ground_truth": "If a complaint is unresolved at the branch level, the customer can escalate it to the Regional Manager, then to the Zonal Manager, then to the Chief Grievance Redressal Officer at Head Office. If unresolved within 30 days, the customer can approach the Banking Ombudsman.",
        "category": "grievance",
    },
    {
        "question": "Can an NRI open a joint account with a resident Indian?",
        "ground_truth": "Yes, an NRI can open a joint account with a resident Indian, but it must be on a 'Former or Survivor' basis only.",
        "category": "nri",
    },
    {
        "question": "What is the maximum loan limit for the Union MSME scheme?",
        "ground_truth": "The Union MSME scheme supports loans up to ₹5 crores for small business working capital or asset purchase without collateral under the CGTMSE coverage.",
        "category": "corporate",
    },
    {
        "question": "What is the current MCLR rate for home loans?",
        "ground_truth": "The marginal cost of funds-based lending rate (MCLR) is reviewed monthly. For home loans, the standard 1-year MCLR rate typically ranges between 8.25% to 8.65% depending on risk premium rating.",
        "category": "interest_rates",
    },
    {
        "question": "How can I activate internet banking for my account?",
        "ground_truth": "Internet banking can be activated online via the Union Bank website by inputting your debit card details and registered mobile number, or by submitting a physical application form at your home branch.",
        "category": "digital",
    },
]


def get_evaluation_dataset() -> list[EvaluationRecord]:
    """Return the golden test dataset wrapped in EvaluationRecord schemas."""
    return [
        EvaluationRecord(
            question=item["question"],
            contexts=[],  # Contexts populated during dynamic test runs
            answer="",    # Answer populated during dynamic test runs
            ground_truth=item["ground_truth"],
        )
        for item in GOLDEN_DATASET
    ]
