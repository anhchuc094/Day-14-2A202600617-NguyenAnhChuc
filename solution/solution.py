"""
Day 14 — AI Evaluation & Benchmarking Pipeline

Completed solution for the lab TODOs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class QAPair:
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    conciseness: float | None = None

    def overall_score(self) -> float:
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "do", "does", "did", "should", "would", "could", "can",
}


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


class RAGASEvaluator:
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        return _clamp(len(answer_tokens & context_tokens) / len(answer_tokens))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        return _clamp(len(answer_tokens & question_tokens) / len(question_tokens))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        return _clamp(len(answer_tokens & expected_tokens) / len(expected_tokens))

    def evaluate_conciseness(
        self,
        answer: str,
        expected: str,
        max_ratio: float = 2.0,
    ) -> float:
        """Custom metric: penalize answers that are much longer than needed.

        This is intentionally separate from overall_score() because it is a
        bonus/business-quality signal, not one of the three required RAGAS-style
        answer metrics. It helps detect verbosity bias in judge workflows.
        """
        answer_tokens = _tokenize(answer)
        expected_tokens = _tokenize(expected)
        if not answer_tokens or not expected_tokens:
            return 1.0

        ratio = len(answer_tokens) / len(expected_tokens)
        if ratio <= max_ratio:
            return 1.0
        return _clamp(max_ratio / ratio)

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens.update(_tokenize(chunk))
        return _clamp(len(expected_tokens & union_tokens) / len(expected_tokens))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevance_flags: list[bool] = []
        for chunk in contexts:
            coverage = len(_tokenize(chunk) & expected_tokens) / len(expected_tokens)
            relevance_flags.append(coverage >= relevance_threshold)

        relevant_total = sum(relevance_flags)
        if relevant_total == 0:
            return 0.0

        relevant_seen = 0
        precision_sum = 0.0
        for rank, relevant in enumerate(relevance_flags, start=1):
            if relevant:
                relevant_seen += 1
                precision_sum += relevant_seen / rank
        return _clamp(precision_sum / relevant_total)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
    ) -> EvalResult:
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)
        conciseness = self.evaluate_conciseness(answer, expected)
        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        return EvalResult(
            qa_pair=QAPair(question=question, expected_answer=expected, context=context),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            conciseness=conciseness,
        )


def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    query_tokens = _tokenize(query)
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


class LLMJudge:
    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        rubric_lines = "\n".join(f"- {name}: {description}" for name, description in rubric.items())
        prompt = (
            "Score the answer from 0 to 1 for each rubric criterion. "
            "Return JSON where keys are criterion names and values are numeric scores.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRubric:\n{rubric_lines}"
        )
        raw = self.judge_llm_fn(prompt)

        try:
            parsed = json.loads(raw)
            scores = {
                key: _clamp(float(parsed.get(key, 0.5)))
                for key in rubric
            }
            if not scores and isinstance(parsed, dict):
                scores = {
                    str(key): _clamp(float(value))
                    for key, value in parsed.items()
                    if isinstance(value, int | float)
                }
        except (TypeError, ValueError, json.JSONDecodeError):
            scores = {key: 0.5 for key in rubric}

        if not scores:
            scores = {key: 0.5 for key in rubric}

        return {"scores": scores, "reasoning": raw}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_scores: list[float] = []
        first_scores: list[float] = []
        other_scores: list[float] = []

        for item in scores_batch:
            scores = item.get("scores", {})
            if isinstance(scores, dict):
                all_scores.extend(float(v) for v in scores.values() if isinstance(v, int | float))

            if "first_score" in item and "second_score" in item:
                first_scores.append(float(item["first_score"]))
                other_scores.append(float(item["second_score"]))
            elif "position" in item and isinstance(scores, dict):
                vals = [float(v) for v in scores.values() if isinstance(v, int | float)]
                if vals and item["position"] == "first":
                    first_scores.append(sum(vals) / len(vals))
                elif vals:
                    other_scores.append(sum(vals) / len(vals))

        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        positional_bias = False
        if first_scores and other_scores:
            positional_bias = (sum(first_scores) / len(first_scores)) > (sum(other_scores) / len(other_scores)) + 0.1

        return {
            "positional_bias": positional_bias,
            "leniency_bias": avg > 0.8,
            "severity_bias": bool(all_scores) and avg < 0.3,
        }


class BenchmarkRunner:
    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
            )
            result.qa_pair = pair
            if pair.retrieved_contexts:
                result.context_recall = evaluator.evaluate_context_recall(
                    pair.retrieved_contexts,
                    pair.expected_answer,
                )
                result.context_precision = evaluator.evaluate_context_precision(
                    pair.retrieved_contexts,
                    pair.expected_answer,
                )
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_conciseness": 0.0,
                "failure_types": {},
            }

        failure_types: dict[str, int] = {}
        for result in results:
            if result.failure_type:
                failure_types[result.failure_type] = failure_types.get(result.failure_type, 0) + 1

        passed = sum(1 for result in results if result.passed)
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": sum(r.faithfulness for r in results) / total,
            "avg_relevance": sum(r.relevance for r in results) / total,
            "avg_completeness": sum(r.completeness for r in results) / total,
            "avg_conciseness": sum(r.conciseness or 0.0 for r in results) / total,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        def avg(results: list[EvalResult], attr: str) -> float:
            return sum(getattr(result, attr) for result in results) / len(results) if results else 0.0

        metrics = ("faithfulness", "relevance", "completeness")
        new_avgs = {metric: avg(new_results, metric) for metric in metrics}
        baseline_avgs = {metric: avg(baseline_results, metric) for metric in metrics}
        regressions = [
            metric
            for metric in metrics
            if baseline_avgs[metric] - new_avgs[metric] > 0.05
        ]

        return {
            "new_avg_faithfulness": new_avgs["faithfulness"],
            "new_avg_relevance": new_avgs["relevance"],
            "new_avg_completeness": new_avgs["completeness"],
            "baseline_avg_faithfulness": baseline_avgs["faithfulness"],
            "baseline_avg_relevance": baseline_avgs["relevance"],
            "baseline_avg_completeness": baseline_avgs["completeness"],
            "regressions": regressions,
            "passed": not regressions,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        return [
            result
            for result in results
            if min(result.faithfulness, result.relevance, result.completeness) < threshold
        ]


class FailureAnalyzer:
    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        categories: dict[str, int] = {}
        for failure in failures:
            failure_type = failure.failure_type or "unknown"
            categories[failure_type] = categories.get(failure_type, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        ordered = sorted(scores.items(), key=lambda item: item[1])
        if len(ordered) > 1 and ordered[1][1] - ordered[0][1] <= 0.05:
            return "Multiple issues detected — review full pipeline"
        lowest = ordered[0][0]
        if lowest == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        if lowest == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        rows = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures, start=1):
            suggestion = suggestions[index - 1] if index - 1 < len(suggestions) else "Review failure and add targeted fix"
            rows.append(
                "| F{index:03d} | {failure_type} | {root_cause} | {suggestion} | Open |".format(
                    index=index,
                    failure_type=self._escape_table(failure.failure_type or "unknown"),
                    root_cause=self._escape_table(self.find_root_cause(failure)),
                    suggestion=self._escape_table(suggestion),
                )
            )
        return "\n".join(rows)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        if categories.get("hallucination", 0) > 0:
            suggestions.append("Add a faithfulness guardrail that rejects claims unsupported by retrieved context")
        if categories.get("irrelevant", 0) > 0 or categories.get("off_topic", 0) > 0:
            suggestions.append("Tighten the prompt with intent checks and require direct answers before elaboration")
        if categories.get("incomplete", 0) > 0:
            suggestions.append("Increase chunk coverage and add few-shot examples that show complete answers")

        if any(f.context_recall is not None and f.context_recall < 0.5 for f in failures):
            suggestions.append("Improve retriever recall with hybrid search, query expansion, or larger top-k")
        if any(f.context_precision is not None and f.context_precision < 0.5 for f in failures):
            suggestions.append("Add reranking or metadata filtering to move relevant chunks ahead of noise")

        fallback = [
            "Review the lowest-scoring examples weekly and add them to the regression dataset",
            "Track score deltas in CI and block deploys when any metric drops by more than 0.05",
            "Calibrate evaluator thresholds against a small human-labeled sample",
        ]
        for item in fallback:
            if len(suggestions) >= 3:
                break
            suggestions.append(item)
        return suggestions

    @staticmethod
    def _escape_table(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG retrieves relevant documents and uses them to ground LLM generation.",
        )
    ]
    runner = BenchmarkRunner()
    evaluator = RAGASEvaluator()
    results = runner.run(qa_pairs, lambda q: "RAG combines retrieval with generation.", evaluator)
    print(runner.generate_report(results))
