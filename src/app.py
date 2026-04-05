import gradio as gr
from document_loader import load_document, chunk_documents
from vector_store import build_vector_store
from hybrid_retriever import hybrid_search
from reranker import rerank
from llm import answer

current_chunks = []

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body, .gradio-container {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1d4ed8, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.main-header p {
    color: #94a3b8;
    font-size: 1rem;
}

.panel {
    background: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}

.status-box {
    background: #0d1117 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
}

button.primary-btn {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}

button.primary-btn:hover { opacity: 0.88 !important; }

.answer-box {
    background: #0d1117 !important;
    border: 1px solid #2563eb !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    color: #e2e8f0 !important;
    min-height: 100px !important;
}

.chunk-card {
    background: #161b27;
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    line-height: 1.75;
    color: #cbd5e1;
}

.chunk-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #0ea5e9;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

.metrics-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}

.metric-badge {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #94a3b8;
}

.metric-badge span {
    color: #6366f1;
    font-weight: 600;
}

label { color: #94a3b8 !important; font-size: 0.8rem !important; font-weight: 500 !important; }

input, textarea {
    background: #0d1117 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

input:focus, textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

.upload-area {
    background: #0d1117 !important;
    border: 2px dashed #2d3148 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}

.upload-area:hover { border-color: #2563eb !important; }

footer { display: none !important; }
"""

def process_document(file):
    global current_chunks
    if file is None:
        return "❌ No file uploaded."
    docs = load_document(file.name)
    current_chunks = chunk_documents(docs)
    build_vector_store(current_chunks)
    return f"✅ Indexed {len(current_chunks)} chunks and ready."

def format_chunks(docs):
    html = ""
    for i, doc in enumerate(docs):
        html += f"""
        <div class='chunk-card'>
            <div class='chunk-label'>Source Chunk {i+1}</div>
            {doc.page_content.strip()}
        </div>
        """
    return html

def ask_question(query):
    global current_chunks
    if not current_chunks:
        return "❌ Please upload and load a document first.", ""
    if not query.strip():
        return "❌ Please enter a question.", ""

    hybrid_results = hybrid_search(query, current_chunks, k=5)
    reranked = rerank(query, hybrid_results, top_n=3)
    response = answer(query, reranked)
    context_html = format_chunks(reranked)
    return response, context_html

HEADER = """
<div class='main-header'>
    <h1>Production RAG Pipeline</h1>
    <p>Hybrid retrieval · Cross-encoder reranking · Local LLM inference · Zero paid APIs</p>
</div>
"""

with gr.Blocks(css=CSS, title="RAG Pipeline") as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### 📄 Document")
            file_input = gr.File(
                label="Upload .txt or .pdf",
                file_types=[".txt", ".pdf"],
                elem_classes="upload-area"
            )
            upload_btn = gr.Button("Load & Index Document", elem_classes="primary-btn")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="status-box"
            )
            upload_btn.click(process_document, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=2, elem_classes="panel"):
            gr.Markdown("### 💬 Ask a Question")
            question_input = gr.Textbox(
                label="",
                placeholder="e.g. What are the risks of AI development?",
                lines=2
            )
            ask_btn = gr.Button("Ask", elem_classes="primary-btn")
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=5,
                elem_classes="answer-box"
            )
            gr.Markdown("#### 📚 Retrieved Context")
            context_output = gr.HTML()
            ask_btn.click(
                ask_question,
                inputs=question_input,
                outputs=[answer_output, context_output]
            )

if __name__ == "__main__":
    demo.launch()