"""
BankAssist RAG — Session Summarizer
====================================
Uses Qwen3-4B to generate structured summaries of finished conversation sessions.

A session summary covers:
  1. Main topics discussed
  2. Products or schemes mentioned (loans, account types)
  3. Customer questions / requests
  4. Banking answers / resolutions given
  5. Action items or unresolved issues remaining
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.llm.qwen3_loader import get_qwen3_model
from app.utils.logger import get_logger

logger = get_logger(__name__)


_SUMMARIZE_SYSTEM_PROMPT = """You are a professional banking assistant.
Summarize the following chat history between a customer and the AI bank assistant.

Generate a structured summary following this format:
- **Topics Discussed**: [Brief list of banking issues or topics]
- **Products Mentioned**: [Specific loan, deposit, card, or digital products]
- **Key Inquiries**: [What did the user ask?]
- **Resolutions Provided**: [Key facts, interest rates, or instructions provided by the assistant]
- **Pending/Action Items**: [Any follow-ups needed, or state "None" if resolved]

Rules:
1. Be extremely concise. Keep the total summary under 150 words.
2. Quote any specific interest rates, amounts, or fee figures exactly as they appear in the history.
3. Output ONLY the structured markdown summary — no introductions, no side talk."""


class SessionSummarizer:
    """
    Summarizes conversation logs at session close to update long-term memory.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = get_qwen3_model()

    def summarize(self, history: list[dict[str, str]]) -> str:
        """
        Generate a structured summary of conversation history using Qwen3.
        """
        if not history:
            return ""

        try:
            logger.info("session_summarization_start", history_length=len(history))
            
            # Format chat history text
            history_lines = []
            for turn in history:
                role = turn.get("role", "user").upper()
                content = turn.get("content", "").strip()
                history_lines.append(f"{role}: {content}")
            
            history_text = "\n".join(history_lines)
            
            prompt = (
                f"<|im_start|>system\n{_SUMMARIZE_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n=== Conversation History ===\n{history_text}\n"
                f"=== Summary ===\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            summary = self._llm.generate_text(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )

            summary_clean = summary.strip()
            
            logger.info(
                "session_summarized_successfully",
                summary_length=len(summary_clean),
            )
            return summary_clean

        except Exception as exc:
            logger.error("failed_to_generate_session_summary", error=str(exc))
            # Fallback simple text summary
            last_inquiry = next(
                (t.get("content", "") for t in reversed(history) if t.get("role") == "user"),
                "unknown inquiry",
            )
            return (
                f"- **Topics Discussed**: General banking inquiry\n"
                f"- **Key Inquiries**: '{last_inquiry[:80]}...'\n"
                f"- **Resolutions Provided**: Answered by assistant\n"
                f"- **Pending/Action Items**: None"
            )
