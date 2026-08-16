# Uterine EMG RAG

A Retrieval-Augmented Generation (RAG) system for uterine electromyography (EMG) research papers. This project aims to build an intelligent question-answering system that retrieves relevant scientific literature and generates accurate, context-aware responses using Large Language Models (LLMs).

---

## Project Structure

```
uterine-emg-rag/

├── notebooks/
│   ├── 01_Module1_Document_Loading.ipynb
│   ├── 02_Load_PDFs.ipynb
│   └── 03_Text_Chunking.ipynb
│
├── papers/
│   ├── *.pdf
│
├── text/
│   ├── *.txt
│
├── processed/
│   ├── chunks.csv
│   └── chunks.json
│
├── images/
│
├── README.md
└── requirements.txt
```

---

## Project Workflow

```
Research Papers (PDF)
          │
          ▼
Module 1: Document Loading
          │
          ▼
Module 2: Text Extraction
          │
          ▼
Module 3: Text Chunking
          │
          ▼
Module 4: Embeddings (Upcoming)
          │
          ▼
Module 5: Vector Database (Upcoming)
          │
          ▼
Module 6: Retrieval (Upcoming)
          │
          ▼
Module 7: LLM Response Generation (Upcoming)
```

---

## Modules Completed

### Module 1 – Document Loading

**Objective**
- Load uterine EMG research papers from PDF format.
- Verify document loading and metadata extraction.

**Tasks**
- Load multiple PDF files.
- Read document metadata.
- Validate successful document ingestion.

---

### Module 2 – Text Extraction

**Objective**
- Extract textual content from each research paper using PyMuPDF.

**Tasks**
- Read all PDF files.
- Extract text page by page.
- Store extracted text as individual `.txt` files.
- Generate basic statistics such as word counts.

**Output**
- One `.txt` file per research paper.

---

### Module 3 – Text Chunking

**Objective**
- Split extracted text into smaller overlapping chunks suitable for semantic search.

**Tasks**
- Read extracted text files.
- Split text using `RecursiveCharacterTextSplitter`.
- Create overlapping chunks.
- Store chunk metadata.

**Output**
- `chunks.csv`
- `chunks.json`

Each chunk contains:
- Source paper
- Chunk ID
- Chunk text

---

## Technologies Used

- Python
- Google Colab
- PyMuPDF
- LangChain
- LangChain Text Splitters
- Pandas

---

## Current Status

- ✅ Module 1 – Completed
- ✅ Module 2 – Completed
- ✅ Module 3 – Completed
- ⏳ Module 4 – Embedding Generation
- ⏳ Module 5 – Vector Database
- ⏳ Module 6 – Semantic Retrieval
- ⏳ Module 7 – Question Answering with LLM

---

## Dataset

Currently includes **10 uterine EMG research papers** in PDF format.

> **Note:** The research papers are not included in this repository due to copyright restrictions.