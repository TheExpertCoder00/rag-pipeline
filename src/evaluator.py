from sentence_transformers import SentenceTransformer, util
from document_loader import load_document, chunk_documents
from hybrid_retriever import hybrid_search
from reranker import rerank
from llm import answer
import json

EVAL_MODEL = "all-MiniLM-L6-v2"

def cosine_score(model, text_a, text_b):
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    return round(float(util.cos_sim(emb_a, emb_b)), 3)

def evaluate_single(query, docs, response, model):
    context = " ".join([doc.page_content for doc in docs])
    faithfulness = cosine_score(model, response, context)
    relevancy = cosine_score(model, query, context)
    coverage = cosine_score(model, query, response)
    return {
        "question": query,
        "answer": response,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "coverage": coverage,
        "avg_score": round((faithfulness + relevancy + coverage) / 3, 3)
    }

def run_eval_suite(test_questions, chunks):
    model = SentenceTransformer(EVAL_MODEL)
    results = []
    for i, query in enumerate(test_questions):
        print(f"Evaluating {i+1}/{len(test_questions)}: {query}")
        hybrid_results = hybrid_search(query, chunks, k=5)
        reranked = rerank(query, hybrid_results, top_n=3)
        response = answer(query, reranked)
        result = evaluate_single(query, reranked, response, model)
        results.append(result)
        print(f"  Faithfulness: {result['faithfulness']} | Relevancy: {result['relevancy']} | Coverage: {result['coverage']} | Avg: {result['avg_score']}")
    return results

if __name__ == "__main__":
    test_questions = [
        "what are the risks of AI development?",
        "how does AI affect employment?",
        "what is the governance gap in AI?",
        "how do great powers compete in AI?",
        "how does AI help programmers?"
    ]
    docs = load_document("data/sample.txt")
    chunks = chunk_documents(docs)
    results = run_eval_suite(test_questions, chunks)
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    avg_overall = round(sum(r["avg_score"] for r in results) / len(results), 3)
    print(f"\n✅ Evaluation complete. Overall avg score: {avg_overall}")
    print("Results saved to data/eval_results.json")