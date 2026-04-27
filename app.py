from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


DEFAULT_VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
TOP_K = 3


@dataclass
class RetrievedChunk:
    score: float
    source: str
    page: int | str
    text: str


@dataclass
class AppConfig:
    vectorstore_dir: Path
    embed_model: str
    llm_model: str
    openai_api_key: str | None


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


def print_sources(results: List[RetrievedChunk]) -> None:
    print("\nSources used:\n")
    if not results:
        print("No sources available.")
        return

    seen: set[tuple[str, int | str]] = set()
    source_rank = 1
    for item in results:
        file_name = Path(item.source).name
        key = (file_name, item.page)
        if key in seen:
            continue
        seen.add(key)
        print(f"{source_rank}. {file_name} (page {item.page})")
        source_rank += 1


def build_context(results: List[RetrievedChunk]) -> str:
    texts = [item.text for item in results if item.text]
    return "\n\n---\n\n".join(texts)


def generate_answer_with_llm(client: OpenAI, llm_model: str, query: str, context: str) -> str:
    system_prompt = (
        "You are a retrieval-augmented assistant. Use only the provided context. "
        "If the answer is not present in the context, respond exactly with: "
        "\"The answer is not available in the provided documents.\" "
        "Keep the answer concise, well-structured, and factual. Do not hallucinate."
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "Answer the question based strictly on this context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{query}"
            ),
        },
    ]
    response = client.chat.completions.create(model=llm_model, messages=messages, temperature=0.0)
    return (response.choices[0].message.content or "").strip()


def run_retrieval(query: str, index: faiss.Index, chunks: List[dict[str, Any]], model: SentenceTransformer) -> List[RetrievedChunk]:
    query_vector = embed_query(query, model)
    return retrieve_top_chunks(index, chunks, query_vector, TOP_K)


def run_rag_answer(query: str, results: List[RetrievedChunk], api_key: str, llm_model: str) -> str:
    context = build_context(results)
    client = OpenAI(api_key=api_key)
    return generate_answer_with_llm(client, llm_model, query, context)


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", str(DEFAULT_VECTORSTORE_DIR))),
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_MODEL_NAME),
        llm_model=os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def main() -> None:
    config = load_config()

    index, chunks = load_vectorstore(config.vectorstore_dir)
    model = SentenceTransformer(config.embed_model)

    query = input("Enter your question: ").strip()
    if not query:
        print("Query is empty. Please enter a valid question.")
        return

    results = run_retrieval(query, index, chunks, model)
    print_results(results)

    if not config.openai_api_key:
        print("OPENAI_API_KEY is not set. Skipping LLM answer generation.")
        return

    answer = run_rag_answer(query, results, config.openai_api_key, config.llm_model)
    print("\nFinal Answer:\n")
    print(answer)
    print_sources(results)


if __name__ == "__main__":
    main()
