
import re
import string
from typing import Sequence


def hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Return 1 if any relevant chunk appears in the top k results."""
    _validate_k(k)

    if not relevant_chunk_ids:
        return 0.0

    relevant = set(relevant_chunk_ids)
    return float(any(chunk_id in relevant for chunk_id in retrieved_chunk_ids[:k]))


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Return the fraction of relevant chunks found in the top k."""
    _validate_k(k)

    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def mean_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> float:
    """Return the reciprocal rank of the first relevant chunk."""
    relevant = set(relevant_chunk_ids)

    if not relevant:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def normalize_answer(answer: str) -> str:
    """Normalize text for basic exact-match evaluation."""
    answer = answer.lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", answer).strip()


def exact_match(
    predicted_answer: str,
    expected_answer: str,
) -> float:
    """Return 1 for a normalized exact match; otherwise 0."""
    return float(
        normalize_answer(predicted_answer)
        == normalize_answer(expected_answer)
    )



def answer_containment(
    predicted_answer: str,
    expected_answer: str,
) -> float:
    """Return 1 if the normalized expected answer appears in the prediction."""
    predicted = normalize_answer(predicted_answer)
    expected = normalize_answer(expected_answer)

    if not expected:
        return float(not predicted)

    return float(expected in predicted)


def answerability_correct(
    predicted_answerable: bool,
    expected_answerable: bool,
) -> float:
    """Check whether the system correctly identifies answerability."""
    return float(predicted_answerable == expected_answerable)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
