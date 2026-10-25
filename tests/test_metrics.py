import pytest

from src.evaluation.metrics import answer_containment


from src.evaluation.metrics import (
    answerability_correct,
    exact_match,
    hit_at_k,
    mean_reciprocal_rank,
    normalize_answer,
    recall_at_k,
)


def test_hit_at_k():
    assert hit_at_k(["c1", "c2", "c3"], ["c3"], k=3) == 1.0
    assert hit_at_k(["c1", "c2"], ["c3"], k=2) == 0.0


def test_recall_at_k():
    assert recall_at_k(
        ["c1", "c2", "c3"], ["c2", "c3", "c4"], k=3
    ) == pytest.approx(2 / 3)


def test_mrr():
    assert mean_reciprocal_rank(
        ["c1", "c2", "c3"], ["c3"]
    ) == pytest.approx(1 / 3)


def test_no_relevant_chunks():
    assert hit_at_k(["c1"], [], k=1) == 0.0
    assert recall_at_k(["c1"], [], k=1) == 0.0
    assert mean_reciprocal_rank(["c1"], []) == 0.0


def test_exact_match():
    assert exact_match("The Answer!", "the answer") == 1.0
    assert exact_match("different answer", "expected answer") == 0.0


def test_normalize_answer():
    assert normalize_answer("  Hello,   WORLD! ") == "hello world"


def test_answerability_correct():
    assert answerability_correct(True, True) == 1.0
    assert answerability_correct(False, True) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        hit_at_k(["c1"], ["c1"], k=k)




def test_answer_containment_with_extra_words():
    assert answer_containment(
        "The separation was completed on July 1, 2026.",
        "July 1, 2026",
    ) == 1.0


def test_answer_containment_with_exact_answer():
    assert answer_containment(
        "$15,336 million",
        "$15,336 million",
    ) == 1.0


def test_answer_containment_rejects_wrong_value():
    assert answer_containment(
        "$8.23 per share",
        "$8.56",
    ) == 0.0
