# Production RAG Pipeline

A production-style Retrieval-Augmented Generation (RAG) system built from scratch in Python. Combines hybrid retrieval, cross-encoder reranking, local LLM inference, and a custom evaluation harness — no paid APIs required.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## Overview

Most RAG demos stop at basic vector search. This project implements the full production stack:

- **Hybrid Retrieval** — BM25 keyword search + FAISS vector similarity, fused with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — a BERT-based reranker re-scores retrieved chunks by reading the query and chunk *together*, not independently
- **Local LLM Inference** — Llama 3.2 via Ollama runs entirely on-device with no API key or internet connection required
- **Custom Evaluation Harness** — faithfulness, relevancy, and coverage scored via cosine similarity across a test suite of questions
- **Gradio UI** — clean dark-themed interface with document upload, Q&A, and source chunk transparency

---

## Architecture

The pipeline runs in 6 stages:

1. **Document Loading** — Accepts `.txt` or `.pdf` files via LangChain loaders
2. **Text Chunking** — Splits documents into 500-character chunks with 100-character overlap using `RecursiveCharacterTextSplitter`
3. **Hybrid Retrieval** — Runs both FAISS vector search (`all-MiniLM-L6-v2`) and BM25 keyword search in parallel, then merges results using Reciprocal Rank Fusion (RRF)
4. **Reranking** — A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the top candidates by reading the query and each chunk together
5. **LLM Inference** — Top 3 reranked chunks are passed as context to Llama 3.2 running locally via Ollama
6. **Grounded Answer** — The LLM responds using only the provided context, refusing to hallucinate if the answer isn't there

---

## Evaluation Results

Evaluated across 5 test questions using a custom harness measuring three metrics (0.0–1.0):

| Question | Faithfulness | Relevancy | Coverage | Avg |
|---|---|---|---|---|
| What are the risks of AI development? | 0.737 | 0.689 | 0.627 | 0.684 |
| How does AI affect employment? | 0.787 | 0.672 | 0.775 | 0.745 |
| What is the governance gap in AI? | 0.641 | 0.666 | 0.892 | 0.733 |
| How do great powers compete in AI? | 0.709 | 0.633 | 0.771 | 0.704 |
| How does AI help programmers? | 0.771 | 0.699 | 0.793 | 0.754 |
| **Overall** | **0.729** | **0.672** | **0.772** | **0.724** |

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Relevancy** — did retrieval surface chunks relevant to the question?
- **Coverage** — does the answer address what was actually asked?

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Keyword Retrieval | BM25 (rank-bm25) |
| Retrieval Fusion | Reciprocal Rank Fusion (custom implementation) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.2 via Ollama |
| UI | Gradio |
| Document Loaders | LangChain (PDF + TXT) |

---
```
## Project Structure

rag-pipeline/
├── src/
│ ├── document_loader.py # PDF/TXT loading and recursive chunking
│ ├── vector_store.py # FAISS index build and load
│ ├── bm25_retriever.py # BM25 keyword retriever
│ ├── hybrid_retriever.py # RRF fusion of BM25 + FAISS
│ ├── reranker.py # Cross-encoder reranking
│ ├── llm.py # Ollama LLM integration
│ ├── evaluator.py # Custom evaluation harness
│ └── app.py # Gradio UI
├── data/
│ ├── sample.txt
│ └── eval_results.json
├── requirements.txt
└── README.md
```
---

## Setup

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com) installed.

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull the LLM**
```bash
ollama pull llama3.2
```

**5. Run the app**
```bash
python src/app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Why Hybrid Retrieval?

Pure vector search understands meaning but can miss exact keywords. Pure BM25 matches keywords but misses paraphrasing. Reciprocal Rank Fusion combines both by merging rank positions rather than raw scores (which live on incompatible scales), giving the best of both approaches with no hyperparameter tuning required.

## Why a Custom Eval Harness?

Most student RAG projects just demo a working answer. This project measures *how well* the pipeline performs across multiple questions using three independent metrics. The harness runs entirely locally using cosine similarity between embeddings — no external eval API, no GPT-4 judge.

---

## License

MIT