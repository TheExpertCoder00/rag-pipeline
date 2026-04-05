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
