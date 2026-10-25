
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.bm25 = BM25Okapi(
            [text.lower().split() for text in texts]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 15,
    ) -> list[dict]:
        if not query.strip():
            return []
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive.")

        k = min(candidate_k, len(self.chunks))

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        _, dense_ids = self.index.search(query_vector, k)

        lexical_scores = self.bm25.get_scores(query.lower().split())
        lexical_ids = np.argsort(lexical_scores)[::-1][:k]

        fused_scores = defaultdict(float)

        # Reciprocal Rank Fusion
        for rank, idx in enumerate(dense_ids[0], start=1):
            if idx >= 0:
                fused_scores[int(idx)] += 1.0 / (60 + rank)

        for rank, idx in enumerate(lexical_ids, start=1):
            fused_scores[int(idx)] += 1.0 / (60 + rank)

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        results = []
        for idx in ranked_ids:
            item = dict(self.chunks[idx])
            item["score"] = float(fused_scores[idx])
            results.append(item)

        return results
