
import json
import statistics
import sys
from pathlib import Path

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_documents
from src.document_loader import load_pdf
from src.evaluation.metrics import (
    answerability_correct,
    answer_containment,
    exact_match,
)
from src.generator import AnswerGenerator
from src.retriever import HybridRetriever


BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation" / "benchmark.json"
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "sec_fill.pdf"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

ABSTENTION_PHRASE = "could not find this information"


def prepare_chunks(pdf_path: Path) -> list[dict]:
    pages = load_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")

    chunks = chunk_documents(pages, chunk_size=900, overlap=150)
    prepared = []

    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        prepared.append({
            "chunk_id": f"chunk_{index:04d}",
            "text": chunk["text"],
            "source": metadata.get("source", pdf_path.name),
            "page": metadata.get("page", "Unknown"),
            "metadata": metadata,
        })

    if not prepared:
        raise ValueError("No chunks were created from the PDF.")

    return prepared


def mean_or_none(values: list[float]) -> float | None:
    """Calculate a mean safely when no values are available."""
    return statistics.mean(values) if values else None


def run_benchmark() -> dict:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    questions = benchmark.get("questions", [])
    if not questions:
        raise ValueError("The benchmark contains no questions.")

    chunks = prepare_chunks(PDF_PATH)
    retriever = HybridRetriever(chunks)
    generator = AnswerGenerator()

    results = []

    for position, item in enumerate(questions, start=1):
        question_id = item["id"]
        question = item["question"]
        relevant_pages = set(item.get("relevant_pages", []))
        expected_answerable = bool(item["answerable"])

        print(f"[{position}/{len(questions)}] {question_id}: {question}")

        retrieved = retriever.search(
            question,
            top_k=5,
            candidate_k=15,
        )

        retrieved_pages = [chunk.get("page") for chunk in retrieved]

        # Page-based retrieval metrics, not chunk-level metrics.
        hit_at_5 = (
            float(any(page in relevant_pages for page in retrieved_pages))
            if relevant_pages
            else None
        )

        if relevant_pages:
            found_pages = set(retrieved_pages) & relevant_pages
            page_recall = len(found_pages) / len(relevant_pages)
        else:
            page_recall = None

        predicted_answer = ""
        citations = []
        generation_error = None
        predicted_answerable = None
        answerability_score = None
        answer_match = None
        answer_containment_score = None

        try:
            generated = generator.generate(question, retrieved)

            predicted_answer = str(generated.get("answer") or "").strip()
            citations = generated.get("citations", [])

            if not predicted_answer:
                generation_error = "Empty model response"
            else:
                # Answerability is based on the model's answer/abstention,
                # not on whether citations were returned.
                is_abstention = (
                    ABSTENTION_PHRASE in predicted_answer.lower()
                )
                predicted_answerable = not is_abstention

                answerability_score = answerability_correct(
                    predicted_answerable,
                    expected_answerable,
                )

                if expected_answerable:
                    answer_match = exact_match(
                        predicted_answer,
                        item["expected_answer"],
                    )
                    answer_containment_score = answer_containment(
                        predicted_answer,
                        item["expected_answer"],
                    )

        except Exception as exc:
            # A failed request is not a model prediction.
            generation_error = f"{type(exc).__name__}: {exc}"

        generation_success = generation_error is None

        results.append({
            "id": question_id,
            "question": question,
            "expected_answerable": expected_answerable,
            "predicted_answerable": predicted_answerable,
            "answerability_correct": answerability_score,
            "expected_answer": item["expected_answer"],
            "predicted_answer": predicted_answer,
            "exact_match": answer_match,
            "answer_containment": answer_containment_score,
            "relevant_pages": sorted(relevant_pages),
            "retrieved_pages": retrieved_pages,
            "page_hit_at_5": hit_at_5,
            "page_recall_at_5": page_recall,
            "retrieved_chunk_ids": [
                chunk["chunk_id"] for chunk in retrieved
            ],
            "citation_indexes": citations,
            "has_citations": bool(citations),
            "generation_success": generation_success,
            "generation_error": generation_error,
        })

        if generation_error:
            print(f"  Generation failed: {generation_error}")
        elif answer_match is not None:
            print(f"  Exact match: {answer_match}")

    answerable_results = [
        row for row in results if row["expected_answerable"]
    ]
    unanswerable_count = sum(
        not row["expected_answerable"] for row in results
    )

    # Only successful, non-empty model responses count as predictions.
    evaluated_answerability = [
        row["answerability_correct"]
        for row in results
        if row["answerability_correct"] is not None
    ]

    successful_answerable = [
        row for row in answerable_results
        if row["generation_success"]
    ]

    exact_match_successful = [
        row["exact_match"] for row in successful_answerable
        if row["exact_match"] is not None
    ]

    containment_successful = [
        row["answer_containment"]
        for row in successful_answerable
        if row["answer_containment"] is not None
    ]

    page_hit_results = [
        row["page_hit_at_5"] for row in results
        if row["page_hit_at_5"] is not None
    ]
    page_recall_results = [
        row["page_recall_at_5"] for row in results
        if row["page_recall_at_5"] is not None
    ]

    successful_count = sum(row["generation_success"] for row in results)
    error_count = len(results) - successful_count

    summary = {
        "benchmark_name": benchmark.get("benchmark_name"),
        "document": str(PDF_PATH.relative_to(PROJECT_ROOT)),
        "total_questions": len(results),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": unanswerable_count,

        "generation_success_count": successful_count,
        "generation_error_count": error_count,
        "generation_success_rate": successful_count / len(results),

        "answerability_evaluated_count": len(evaluated_answerability),
        "answerability_accuracy": mean_or_none(evaluated_answerability),

        "answerable_generation_count": len(successful_answerable),
        "answerable_generation_coverage": (
            len(successful_answerable) / len(answerable_results)
            if answerable_results else None
        ),
        "exact_match_on_successful_answerable_generations": (
            mean_or_none(exact_match_successful)
        ),

        "answer_containment_on_successful_answerable_generations": (
            mean_or_none(containment_successful)
        ),

        # Failed generations count as not exact matches for this metric.
        "exact_match_over_all_answerable_questions": (
            sum(
                row["exact_match"] == 1
                for row in answerable_results
                if row["exact_match"] is not None
            )
            / len(answerable_results)
            if answerable_results else None
        ),

        "page_hit_at_5": mean_or_none(page_hit_results),
        "page_recall_at_5": mean_or_none(page_recall_results),

        "note": (
            "Answerability accuracy excludes generation failures. "
            "Answerable generation coverage is the fraction of all "
            "answerable questions with a successful, non-empty response. "
            "Exact match is strict and does not capture semantically "
            "correct paraphrases. The all-answerable exact-match metric "
            "counts failed generations as non-matches. Retrieval metrics "
            "are page-based proxies, not chunk-level Hit@K, Recall@K, or MRR. "
            "Answerability is inferred from the model's refusal wording; "
            "citation presence is tracked separately and does not prove "
            "that an answer is correct or supported."
        ),
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "benchmark_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nBenchmark complete.")
    print(f"Total questions: {summary['total_questions']}")
    print(
        f"Generation success rate: "
        f"{summary['generation_success_rate']:.1%}"
    )
    print(f"Generation errors: {summary['generation_error_count']}")
    print(
        f"Answerability accuracy "
        f"(successful responses only): "
        f"{summary['answerability_accuracy']}"
    )
    print(
        f"Answerable generation coverage: "
        f"{summary['answerable_generation_coverage']}"
    )
    print(
        f"Exact match (successful answerable generations): "
        f"{summary['exact_match_on_successful_answerable_generations']}"
    )
    print(
        f"Exact match (all answerable questions): "
        f"{summary['exact_match_over_all_answerable_questions']}"
    )
    print(
        "Answer containment (successful answerable generations): "
        f"{summary['answer_containment_on_successful_answerable_generations']}"
    )
    print(f"Page Hit@5: {summary['page_hit_at_5']}")
    print(f"Page Recall@5: {summary['page_recall_at_5']}")
    print(f"Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_benchmark()
