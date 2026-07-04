"""Quick checks for evaluation metrics and question loading."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.evaluation.evaluator import load_eval_questions
from src.evaluation.metrics import document_hit_at_k, factual_accuracy_score


def main() -> None:
    settings = load_settings()
    questions = load_eval_questions(settings)
    print(f"Eval questions loaded: {len(questions)}")
    for item in questions:
        print(f"  - {item.reference_documents[0] if item.reference_documents else '?'} | year={item.year} period={item.period_code}")

    assert factual_accuracy_score("<FINAL_ANSWER>23918635</FINAL_ANSWER>", "$23,918,635")
    assert factual_accuracy_score("<FINAL_ANSWER>4.815</FINAL_ANSWER>", "4.815")
    assert factual_accuracy_score("<FINAL_ANSWER>4.815%</FINAL_ANSWER>", "4.815")
    print("Factual accuracy unit checks passed.")


if __name__ == "__main__":
    main()
