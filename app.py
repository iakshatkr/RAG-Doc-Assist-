from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


@dataclass
class RetrievedChunk:
    score: float
    source: str
    page: int | str
    text: str


def load_vectorstore(vectorstore_dir: Path) -> tuple[faiss.Index, List[dict[str, Any]]]:
    index_path = vectorstore_dir / "index.faiss"
    chunks_path = vectorstore_dir / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunk objects.")
    return index, chunks


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.astype("float32")


def search(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(query_vector, top_k)


def retrieve_top_chunks(
    index: faiss.Index, chunks: List[dict[str, Any]], query_vector: np.ndarray, top_k: int
) -> List[RetrievedChunk]:
    scores, indices = search(index, query_vector, top_k)
    results: List[RetrievedChunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                score=float(score),
                source=str(chunk.get("source", "unknown")),
                page=chunk.get("page", "?"),
                text=str(chunk.get("text", "")).strip(),
            )
        )
    return results


def print_results(results: List[RetrievedChunk]) -> None:
    print("\nTop 3 similar chunks:\n")
    if not results:
        print("No matching chunks found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"{rank}. Score: {item.score:.4f}")
        print(f"   Source: {item.source} (page {item.page})")
        print(f"   Text: {item.text}\n")


def build_context(results: List[RetrievedChunk]) -> str:
    texts = [item.text for item in results if item.text]
    return "\n\n---\n\n".join(texts)


def main() -> None:
    vectorstore_dir = Path(os.getenv("VECTORSTORE_DIR", str(DEFAULT_VECTORSTORE_DIR)))
    model_name = os.getenv("EMBED_MODEL", DEFAULT_MODEL_NAME)

    index, chunks = load_vectorstore(vectorstore_dir)
    model = SentenceTransformer(model_name)

    query = input("Enter your question: ").strip()
    if not query:
        print("Query is empty. Please enter a valid question.")
        return

    query_vector = embed_query(query, model)
    results = retrieve_top_chunks(index, chunks, query_vector, TOP_K)
    print_results(results)
    _context = build_context(results)


if __name__ == "__main__":
    main()
