# Production RAG Pipeline with Hybrid Retrieval

A production-style Retrieval-Augmented Generation (RAG) system built with hybrid retrieval (BM25 + vector search), cross-encoder reranking, and a custom evaluation harness.

## Stack
- `sentence-transformers` — text embeddings
- `FAISS` — vector similarity search
- `BM25` — keyword retrieval
- `Ollama` — local LLM inference (no API key needed)
- `Gradio` — demo UI

## Progress
- [x] Project setup
- [ ] Document loading + chunking
- [ ] Embeddings + vector store
- [ ] Hybrid retrieval
- [ ] Reranking
- [ ] LLM integration
- [ ] Evaluation harness
- [ ] UI