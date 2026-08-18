import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a given PDF file."""
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return text

def main():
    # 1. Define paths
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load PDFs
    print(f"Scanning for PDFs in {raw_dir}...")
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found in data/raw/. Please add your research papers.")
        return

    print(f"Found {len(pdf_files)} PDFs. Extracting text...")
    
    papers = {}
    for pdf in pdf_files:
        papers[pdf.stem] = extract_text_from_pdf(pdf)

    # 3. Chunk the text
    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    for paper_name, text in papers.items():
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "paper": paper_name,
                "chunk_id": i,
                "text": chunk,
                "characters": len(chunk)
            })

    # Save metadata
    metadata_df = pd.DataFrame(all_chunks)
    metadata_path = processed_dir / "chunk_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata for {len(metadata_df)} chunks to {metadata_path}")

    # 4. Generate Embeddings
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Generating embeddings (this may take a moment on CPU)...")
    texts = metadata_df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Format embeddings for FAISS (float32 is required)
    embeddings = embeddings.astype("float32")
    
    # Save raw embeddings just in case
    np.save(processed_dir / "embeddings.npy", embeddings)

    # 5. Build FAISS Index
    print("Building FAISS index...")
    faiss.normalize_L2(embeddings) # Normalize for Inner Product (Cosine Similarity)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save FAISS index
    index_path = processed_dir / "uterine_emg.index"
    faiss.write_index(index, str(index_path))
    print(f"Successfully saved FAISS index to {index_path}")
    print("Ingestion complete! Pipeline is ready.")

if __name__ == "__main__":
    main()