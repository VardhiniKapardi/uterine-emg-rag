# src/retriever.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

class RAGRetriever:
    def __init__(self, processed_dir: str):
        print("Loading embedding models and FAISS index...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.index = faiss.read_index(f"{processed_dir}/uterine_emg.index")
        self.metadata = pd.read_csv(f"{processed_dir}/chunk_metadata.csv")

    def retrieve_and_rerank(self, query: str, candidate_k=20, final_k=5) -> list:
        # 1. FAISS Dense Retrieval
        query_embedding = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, candidate_k)
        
        candidates = []
        for rank, idx in enumerate(indices[0]):
            candidates.append({
                "text": self.metadata.iloc[idx]["text"],
                "paper": self.metadata.iloc[idx]["paper"],
                "chunk_id": self.metadata.iloc[idx]["chunk_id"]
            })
            
        # 2. CrossEncoder Reranking
        pairs = [[query, doc["text"]] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for doc, score in zip(candidates, rerank_scores):
            doc["rerank_score"] = float(score)
            
        # Sort and return top final_k
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]