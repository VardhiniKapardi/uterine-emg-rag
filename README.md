# Uterine EMG RAG: Domain-Specific Biomedical Literature Assistant

An end-to-end, fully local, CPU-optimized Retrieval-Augmented Generation (RAG) pipeline engineered for **Uterine Electromyography (EMG / Electrohysterography - EHG) research**.

This system automates the retrieval of open-access biomedical literature, indexes semantic document chunks into a dense vector space, applies cross-encoder reranking to prioritize high-fidelity context, and generates grounded, cited responses using a quantized Qwen2.5-3B-Instruct large language model.

---

## 📌 Architecture Overview

```text
                          ┌───────────────────────────┐
                          │   OpenAlex API Search     │
                          │   (Automated OA Discovery)│
                          └─────────────┬─────────────┘
                                        │ (PDFs)
                                        ▼
                          ┌───────────────────────────┐
                          │   Document Preprocessing  │
                          │   - PyMuPDF Extraction    │
                          │   - Recursive Chunking    │
                          └─────────────┬─────────────┘
                                        │ (833 Chunks)
                                        ▼
┌──────────────────┐      ┌───────────────────────────┐
│   User Query     │─────▶│ all-MiniLM-L6-v2 Encoder  │
└──────────────────┘      └─────────────┬─────────────┘
         │                              │ (Dense Embeddings: d=384)
         │                              ▼
         │                ┌───────────────────────────┐
         │                │     FAISS FlatIP Index    │
         │                │   (Cosine Sim Candidate)  │
         │                └─────────────┬─────────────┘
         │                              │ (Top-20 Candidates)
         ▼                              ▼
┌─────────────────────────────────────────────────────┐
│      ms-marco-MiniLM-L-6-v2 CrossEncoder            │
│         (Contextual Score Reranking)                │
└──────────────────────────┬──────────────────────────┘
                           │ (Top-5 Reranked Chunks + Provenance)
                           ▼
┌─────────────────────────────────────────────────────┐
│           Qwen2.5-3B-Instruct (GGUF 4-bit)          │
│           - llama-cpp-python CPU Engine             │
│           - Citation-Aware Prompt Template          │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│          Interactive Streamlit Web Interface        │
│          (Grounded Answer + Reranked Sources)       │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### Automated Paper Acquisition

* Query-driven discovery of open-access EHG/uterine EMG research via the OpenAlex API with automated metadata sanitization and PDF downloading.

### Two-Stage Information Retrieval

**Dense Retrieval:**
Sub-millisecond vector search across 384-dimensional normalized embeddings using FAISS (`IndexFlatIP`).

**Neural Reranking:**
Deep cross-attentional scoring with `ms-marco-MiniLM-L-6-v2` to eliminate irrelevant context prior to generation.

### 100% Local CPU Execution

Quantized 4-bit GGUF inference powered by `llama-cpp-python`, allowing fast generation on standard laptop/desktop CPUs without dedicated GPU hardware.

### Source Attribution & Hallucination Guardrails

Prompt-engineered citation binding `[Source N]` ensuring that factual statements directly map to verified literature excerpts.

### Gold-Standard Evaluation Suite

Benchmarked on **30 domain-specific queries** evaluating retrieval accuracy (**Hit Rate@5, NDCG@5, MRR, Precision@5, and Recall@5**).

---

## 📂 Repository Structure

```text
uterine-emg-rag/
├── .streamlit/
│   └── config.toml           # Streamlit watcher blacklist & UI configuration
├── data/
│   ├── raw/                  # Source research PDFs
│   └── processed/            # FAISS index, chunk metadata, embeddings
├── src/
│   ├── __init__.py           # Package marker
│   ├── search_papers.py      # Automated OpenAlex open-access paper downloader
│   ├── ingest.py             # PDF text extraction, chunking, and FAISS indexing
│   ├── retriever.py          # Vector retrieval and CrossEncoder reranking module
│   └── generator.py          # Quantized Qwen2.5 local CPU generation engine
├── app.py                    # Streamlit web application frontend
├── main.py                   # CLI end-to-end execution pipeline
├── requirements.txt          # Pinned dependency requirements
└── README.md
```

---

# 🚀 Getting Started

## 1. Prerequisites

* Python 3.10+
* Git

---

## 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/VardhiniKapardi/uterine-emg-rag.git
cd uterine-emg-rag

# Create and activate virtual environment
python -m venv ehg_rag_venv

source ehg_rag_venv/bin/activate  # On Windows:
# ehg_rag_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Pipeline Ingestion

**Optional if the index already exists.**

To fetch open-access papers and rebuild the vector database:

```bash
# 1. Download open-access papers from OpenAlex
python src/search_papers.py

# 2. Extract text, generate embeddings, and build the FAISS index
python src/ingest.py
```

---

# 💻 Usage

## (a) Run via Command Line Interface (CLI)

```bash
python main.py
```

## (b) Run the Interactive Web UI

Launch the Streamlit dashboard to test domain queries interactively:

```bash
streamlit run app.py
```

Navigate to:

```text
http://localhost:8501
```

in your browser.

---

# 🔬 Technical Methodology

## 1. Chunking Strategy

Research documents are parsed into structured text and segmented using LangChain's `RecursiveCharacterTextSplitter` with a window size of **1000 characters** and **200 characters overlap**, preserving technical descriptions across sentence boundaries.

## 2. Semantic Indexing & Normalization

Embeddings are computed using `all-MiniLM-L6-v2`. Vectors are explicitly mapped to unit length (L2 normalization) to allow cosine similarity computations via inner product operations (`IndexFlatIP`).

## 3. Neural Reranking

The retriever pulls candidate chunks (**k=20**) via FAISS and feeds `(query, document)` pairs to the `ms-marco-MiniLM-L-6-v2` CrossEncoder. The cross-attention scores reorder the chunks, supplying only the **top k=5** most relevant passages to the LLM.

## 4. Quantized CPU Generation

Answer synthesis uses **Qwen2.5-3B-Instruct** in 4-bit **Q4_K_M GGUF** quantization executed through `llama-cpp-python`. This allows inference on commodity CPUs with low RAM footprint (~2.2 GB) while preventing hallucination via explicit grounding prompts.

---

## 📊 Evaluation & Benchmark Results
The retrieval component was evaluated against a curated gold dataset of **30 specialized clinical and technical questions** covering uterine EMG characteristics, frequency analysis, and preterm birth prediction.

Retrieval Performance

Retrieval Metric      Result
--------------------------------
Hit Rate@5            100.00%
NDCG@5                79.64%
MRR                   1.0000
Precision@5           60.00%
Recall@5              71.11%

These results indicate that the retrieval pipeline successfully returned at least one relevant passage within the top five results for all 30 evaluation queries, while achieving an MRR of 1.0, indicating that the relevant result was consistently ranked first under the evaluation setup.

## ⚡ System Performance
The complete RAG pipeline was also benchmarked for inference performance on a CPU-based environment.

System Metric                      Result
-----------------------------------------------
Open-access papers indexed             41
Semantic chunks indexed                833
Retrieval latency                      0.648 s
End-to-end query latency               10.70 s
LLM generation speed                   21.4 tokens/s
Total RAM usage                        ~4.45 GB

The retrieval stage completes in approximately 648 ms, while the complete query-to-answer pipeline takes approximately 10.7 seconds, including retrieval, reranking, and local LLM generation.
The system therefore provides a practical fully local RAG workflow without requiring a dedicated GPU.

## 🧪 Example Query
Query:
What are the characteristics of uterine EMG signals?

Generated response:
Uterine EMG (Electrohysterography) measures the electrical activity of the uterus, specifically myometrial electrical activation. It captures low-frequency electrical bursts associated with uterine activity and provides a noninvasive representation of the uterine depolarization field. Unlike intrauterine pressure measurements, uterine EMG can be recorded noninvasively and can be used for monitoring uterine activity in both clinical and home settings.
The generated response is accompanied by retrieved literature sources and citation bindings to maintain traceability to the underlying biomedical literature.

# 📜 License

This project is licensed under the **MIT License**.
