from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

LLM_MODEL = "llama3.2"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the context doesn't contain enough information, say "I don't have enough context to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:
"""

def get_llm():
    return OllamaLLM(model=LLM_MODEL)

def build_prompt(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt.format(context=context, question=query)

def answer(query, docs):
    llm = get_llm()
    prompt = build_prompt(query, docs)
    return llm.invoke(prompt)

if __name__ == "__main__":
    from document_loader import load_document, chunk_documents
    from hybrid_retriever import hybrid_search
    from reranker import rerank

    docs = load_document("data/sample.txt")
    chunks = chunk_documents(docs)
    query = "what are the risks of AI development?" #replace with your query
    #is AI ethical wasn't in the provided sample text, so the query is changed here.
    hybrid_results = hybrid_search(query, chunks, k=5)
    reranked = rerank(query, hybrid_results, top_n=3)
    response = answer(query, reranked)
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {response}")