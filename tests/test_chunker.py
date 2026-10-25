
import pytest

from src.chunker import chunk_documents


def test_chunker_splits_text_and_preserves_metadata():
    documents = [
        {
            "text": "abcdefghij",
            "metadata": {"source": "sample.pdf", "page": 1},
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=6,
        overlap=2,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]
    assert chunks[0]["metadata"]["source"] == "sample.pdf"
    assert chunks[0]["metadata"]["page"] == 1


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_documents(
            [{"text": "some text"}],
            chunk_size=4,
            overlap=4,
        )


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_documents(
            [{"text": "some text"}],
            chunk_size=0,
            overlap=0,
        )
