from langchain_community.retrievers import BM25Retriever
from document_loader import load_document, chunk_documents

BM25_RETRIEVER_PATH = "data/bm25_chunks.pkl"

def build_bm25_retriever(chunks, k=3):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever

if __name__ == "__main__":
    docs = load_document("data/sample.txt")
    chunks = chunk_documents(docs)
    retriever = build_bm25_retriever(chunks)
    query = "is AI ethical?" #replace with your query
    results = retriever.invoke(query)
    print(f"Top {len(results)} BM25 results for: '{query}'")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---\n{r.page_content}")