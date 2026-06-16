"""CI quality gate for the Day 14 evaluation pipeline.

The script is intentionally dependency-free so it can run in GitHub Actions
without installing extra packages. It exits with a non-zero status when the
benchmark violates the configured thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution.solution import BenchmarkRunner, QAPair, RAGASEvaluator


MIN_PASS_RATE = 0.60
MIN_FAITHFULNESS = 0.70
MIN_RELEVANCE = 0.65
MIN_COMPLETENESS = 0.60
MIN_CONCISENESS = 0.70


def build_ci_dataset() -> list[QAPair]:
    return [
        QAPair(
            question="What does RAG stand for?",
            expected_answer="RAG stands for Retrieval-Augmented Generation.",
            context="RAG stands for Retrieval-Augmented Generation and combines document retrieval with generation.",
        ),
        QAPair(
            question="What is faithfulness?",
            expected_answer="Faithfulness measures whether answer claims are supported by the provided context.",
            context="Faithfulness measures whether answer claims are supported by the provided context.",
        ),
        QAPair(
            question="How does reranking improve context precision?",
            expected_answer="Reranking moves relevant chunks earlier, improving rank-aware context precision.",
            context="Reranking moves relevant chunks earlier, improving rank-aware context precision.",
        ),
    ]


def agent(question: str) -> str:
    answers = {
        "What does RAG stand for?": "RAG stands for Retrieval-Augmented Generation.",
        "What is faithfulness?": "Faithfulness measures whether answer claims are supported by the context.",
        "How does reranking improve context precision?": (
            "Reranking moves relevant chunks earlier, improving rank-aware context precision."
        ),
    }
    return answers[question]


def main() -> int:
    runner = BenchmarkRunner()
    evaluator = RAGASEvaluator()
    results = runner.run(build_ci_dataset(), agent, evaluator)
    report = runner.generate_report(results)

    checks = {
        "pass_rate": report["pass_rate"] >= MIN_PASS_RATE,
        "faithfulness": report["avg_faithfulness"] >= MIN_FAITHFULNESS,
        "relevance": report["avg_relevance"] >= MIN_RELEVANCE,
        "completeness": report["avg_completeness"] >= MIN_COMPLETENESS,
        "conciseness": report["avg_conciseness"] >= MIN_CONCISENESS,
    }

    print("Evaluation quality gate")
    for key, value in report.items():
        print(f"{key}: {value}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"FAILED checks: {', '.join(failed)}")
        return 1

    print("PASSED all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
