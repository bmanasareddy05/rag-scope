
def chunk_documents(
    documents: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """Split documents into overlapping text chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        text = document["text"].strip()
        metadata = dict(document.get("metadata", {}))

        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()

            if piece:
                chunks.append({
                    "text": piece,
                    "metadata": metadata.copy(),
                })

            if start + chunk_size >= len(text):
                break

    return chunks
