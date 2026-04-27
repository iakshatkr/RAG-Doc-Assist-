# RAG-Doc-Assist-

RAG document assistant with:
- PDF ingestion and chunking
- Sentence-transformer embeddings
- Local FAISS vector store
- Top-k retrieval + LLM grounded answer generation

## Setup

1. Install dependencies:
   `pip install -r requirements.txt`
2. Create `.env` from `.env.example` and set your API key.
3. Ingest documents:
   `python ingest.py`
4. Ask questions:
   `python app.py`
