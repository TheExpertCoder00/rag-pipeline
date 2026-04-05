from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

def rerank(query, docs, top_n=3):
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]

if __name__ == "__main__":
    from document_loader import load_document, chunk_documents
    from hybrid_retriever import hybrid_search

    docs = load_document("data/sample.txt")
    chunks = chunk_documents(docs)
    query = "is AI ethical?" #replace with your query

    print("--- Before Reranking (Hybrid) ---")
    hybrid_results = hybrid_search(query, chunks, k=5)
    for i, r in enumerate(hybrid_results):
        print(f"\nResult {i+1}: {r.page_content[:100]}...")

    print("\n--- After Reranking (Top 3) ---")
    reranked = rerank(query, hybrid_results, top_n=3)
    for i, r in enumerate(reranked):
        print(f"\nResult {i+1}: {r.page_content[:100]}...")