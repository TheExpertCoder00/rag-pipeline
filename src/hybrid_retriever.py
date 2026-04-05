from vector_store import load_vector_store
from bm25_retriever import build_bm25_retriever
from document_loader import load_document, chunk_documents

def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    contents = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content.strip()
            if key not in scores:
                scores[key] = 0
                contents[key] = doc
            scores[key] += 1 / (k + rank + 1)
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [contents[key] for key in sorted_keys]

def hybrid_search(query, chunks, k=5):
    vector_store = load_vector_store()
    faiss_results = vector_store.similarity_search(query, k=k)
    bm25_retriever = build_bm25_retriever(chunks, k=k)
    bm25_results = bm25_retriever.invoke(query)
    fused = reciprocal_rank_fusion([faiss_results, bm25_results])
    return fused[:k]

if __name__ == "__main__":
    docs = load_document("data/sample.txt")
    chunks = chunk_documents(docs)
    query = "is AI ethical?" #replace with your query
    results = hybrid_search(query, chunks)
    print(f"Hybrid search results for: '{query}'")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---\n{r.page_content}")