# src/app.py
import time
import streamlit as st

from src.retriever import RAGRetriever
from src.generator import LocalQwenGenerator

# Cache the heavy models so they don't reload on every keystroke
@st.cache_resource
def load_components():
    retriever = RAGRetriever(processed_dir="./data/processed")
    generator = LocalQwenGenerator()
    return retriever, generator

st.set_page_config(page_title="Uterine EMG RAG", layout="wide")
st.title("🔬 Uterine EMG Research Assistant")
st.markdown("Ask domain-specific questions based on the indexed clinical literature.")

# Initialize pipeline
retriever, generator = load_components()

query = st.text_input("Enter your research query:", "What are the characteristics of uterine EMG signals?")

if st.button("Generate Answer"):
    total_start = time.perf_counter()
    
    with st.spinner("Retrieving and reranking FAISS chunks..."):
        retrieval_output = retriever.retrieve_and_rerank(query, candidate_k=20, final_k=5)
        top_chunks = retrieval_output["results"]
        retrieval_metrics = retrieval_output["metrics"]
        
        context = "\n\n".join([
            f"[Source: {chunk['paper']}] {chunk['text']}" 
            for chunk in top_chunks
        ])
        
    with st.spinner("Generating answer on local CPU..."):
        llm_output = generator.generate_answer(query, context)
        answer = llm_output["content"]
        llm_metrics = llm_output["metrics"]
        
    end_to_end_time = time.perf_counter() - total_start
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### ⚡ System Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("End-to-End Latency", f"{end_to_end_time:.2f} s")
    cols[1].metric("LLM Generation", f"{llm_metrics['tokens_per_sec']} tok/s")
    cols[2].metric("RAM Usage (Total)", f"{llm_metrics['approx_ram_mb']} MB")
    cols[3].metric("Retrieval Latency", f"{retrieval_metrics['total_retrieval_sec']:.3f} s")
    
    st.markdown("### Retrieved Sources (CrossEncoder Reranked)")
    for i, chunk in enumerate(top_chunks):
        with st.expander(f"Source {i+1}: {chunk['paper']} (Score: {chunk['rerank_score']:.2f})"):
            st.write(chunk['text'])