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
    with st.spinner("Retrieving and reranking FAISS chunks..."):
        top_chunks = retriever.retrieve_and_rerank(query, candidate_k=20, final_k=5)
        
        context = "\n\n".join([
            f"[Source: {chunk['paper']}] {chunk['text']}" 
            for chunk in top_chunks
        ])
        
    with st.spinner("Generating answer on local CPU..."):
        answer = generator.generate_answer(query, context)
        
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### Retrieved Sources (CrossEncoder Reranked)")
    for i, chunk in enumerate(top_chunks):
        with st.expander(f"Source {i+1}: {chunk['paper']} (Score: {chunk['rerank_score']:.2f})"):
            st.write(chunk['text'])