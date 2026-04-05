from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_STORE_PATH = "data/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

if __name__ == "__main__":
    from document_loader import load_document, chunk_documents
    docs = load_document("data/sample.txt") #just a sample document about AI
    chunks = chunk_documents(docs)
    vector_store = build_vector_store(chunks)
    query = "is AI ethical?" #replace with your query
    results = vector_store.similarity_search(query, k=3)
    print(f"\nTop 3 results for: '{query}'")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---\n{r.page_content}")