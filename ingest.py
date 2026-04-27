from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import re
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: int
    source: str
    page: int
    text: str


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_DATA_DIR = Path("/data")
FALLBACK_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "vectorstore"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_data_dir() -> Path:
    data_dir_from_env = os.getenv("DATA_DIR")
    if data_dir_from_env:
        return Path(data_dir_from_env)
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    return FALLBACK_DATA_DIR


def load_pdf_texts(data_dir: Path) -> List[ChunkRecord]:
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    records: List[ChunkRecord] = []
    chunk_id = 0

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            clean_text = " ".join(raw_text.split())
            if not clean_text:
                continue
            for chunk in chunk_text(clean_text, CHUNK_SIZE, CHUNK_OVERLAP):
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source=str(pdf_path),
                        page=page_index,
                        text=chunk,
                    )
                )
                chunk_id += 1

    if not records:
        raise ValueError("No extractable text was found in the PDFs.")
    return records


def chunk_text(text: str, size: int, overlap: int) -> Iterable[str]:
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    words = text.split()
    if not words:
        return

    i = 0
    n = len(words)
    while i < n:
        # Build one chunk up to ~size characters while keeping whole words.
        chunk_words: List[str] = []
        chunk_len = 0
        j = i
        while j < n:
            word = words[j]
            candidate_len = chunk_len + (1 if chunk_words else 0) + len(word)
            if chunk_words and candidate_len > size:
                break
            chunk_words.append(word)
            chunk_len = candidate_len
            j += 1

        if not chunk_words:
            # Handle edge case: a single token longer than chunk size.
            chunk_words = [words[i]]
            j = i + 1

        yield " ".join(chunk_words)

        if j >= n:
            break

        # Move start forward but keep ~overlap characters from the end.
        overlap_chars = 0
        back = len(chunk_words) - 1
        while back >= 0 and overlap_chars < overlap:
            overlap_chars += len(chunk_words[back]) + (1 if overlap_chars > 0 else 0)
            back -= 1
        i = max(i + 1, i + back + 1)


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def save_artifacts(index: faiss.Index, records: List[ChunkRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = [
        {
            "chunk_id": r.chunk_id,
            "source": r.source,
            "page": r.page,
            "text": r.text,
        }
        for r in records
    ]
    with (output_dir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    data_dir = resolve_data_dir()
    output_dir = Path(os.getenv("VECTORSTORE_DIR", str(DEFAULT_OUTPUT_DIR)))
    model_name = os.getenv("EMBED_MODEL", DEFAULT_MODEL_NAME)

    print(f"Loading PDFs from: {data_dir}")
    records = load_pdf_texts(data_dir)
    texts = [r.text for r in records]
    print(f"Created {len(texts)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print(f"Generating embeddings with: {model_name}")
    vectors = embed_texts(texts, model_name)

    print("Building FAISS index...")
    index = build_faiss_index(vectors)
    save_artifacts(index, records, output_dir)

    print(f"Saved FAISS index and metadata to: {output_dir}")
    print("Files created:")
    print(f"- {output_dir / 'index.faiss'}")
    print(f"- {output_dir / 'chunks.json'}")


if __name__ == "__main__":
    main()

def clean_text(text):
    # Add space between lowercase and uppercase (common PDF issue)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Add space between words and numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    return text
