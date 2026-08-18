import json
import numpy as np
from pathlib import Path

# Adjust the import based on how you run the script (e.g., from the root directory)
try:
    from src.retriever import RAGRetriever
except ModuleNotFoundError:
    from retriever import RAGRetriever

def normalize_name(name: str) -> str:
    """Normalizes paper names to match accurately (e.g. 'paper1.pdf' -> 'paper1')."""
    return Path(str(name)).stem.lower().strip()

def get_unique_papers(retrieved_chunks: list) -> list:
    """Extracts an ordered list of unique papers from the retrieved chunks."""
    seen = set()
    papers = []
    for chunk in retrieved_chunks:
        paper = chunk["paper"]
        if paper not in seen:
            seen.add(paper)
            papers.append(paper)
    return papers

# ==========================================
# Evaluation Metrics
# ==========================================

def hit_rate_at_k(retrieved_papers: list, relevant_papers: list, k: int = 5) -> int:
    retrieved = retrieved_papers[:k]
    relevant = {normalize_name(p) for p in relevant_papers}
    return int(any(normalize_name(p) in relevant for p in retrieved))

def ndcg_at_k(retrieved_papers: list, relevant_papers: list, k: int = 5) -> float:
    retrieved = retrieved_papers[:k]
    relevant = {normalize_name(p) for p in relevant_papers}
    
    # Binary relevance: 1 if the paper is in the relevant set, else 0
    relevance = [1 if normalize_name(p) in relevant else 0 for p in retrieved]
    
    dcg = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            dcg += rel / np.log2(rank + 1)
            
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    
    return 0.0 if idcg == 0 else dcg / idcg

def precision_at_k(retrieved_papers: list, relevant_papers: list, k: int = 5) -> float:
    retrieved = retrieved_papers[:k]
    relevant = {normalize_name(p) for p in relevant_papers}
    hits = sum(1 for p in retrieved if normalize_name(p) in relevant)
    return hits / k

def recall_at_k(retrieved_papers: list, relevant_papers: list, k: int = 5) -> float:
    retrieved = retrieved_papers[:k]
    relevant = {normalize_name(p) for p in relevant_papers}
    retrieved_set = {normalize_name(p) for p in retrieved}
    
    if not relevant:
        return 0.0
        
    hits = len(retrieved_set.intersection(relevant))
    return hits / len(relevant)

def reciprocal_rank(retrieved_papers: list, relevant_papers: list) -> float:
    relevant = {normalize_name(p) for p in relevant_papers}
    for rank, p in enumerate(retrieved_papers, start=1):
        if normalize_name(p) in relevant:
            return 1.0 / rank
    return 0.0

# ==========================================
# Main Evaluation Loop
# ==========================================

def run_evaluation(golden_queries_path: str, processed_dir: str):
    print(f"Loading queries from {golden_queries_path}...")
    with open(golden_queries_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    retriever = RAGRetriever(processed_dir=processed_dir)
    
    metrics = {
        "Hit Rate@5": [],
        "NDCG@5": [],
        "MRR": [],
        "Precision@5": [],
        "Recall@5": []
    }
    
    print(f"Evaluating {len(gold_data)} queries...\n")
    
    for item in gold_data:
        # Handle variations in JSON keys ("query" vs "question", "relevant_papers" vs "relevant_paper_ids")
        query = item.get("question", item.get("query", ""))
        relevant_papers = item.get("relevant_paper_ids", item.get("relevant_papers", []))
        
        # 1. Retrieve & Rerank
        retrieval_output = retriever.retrieve_and_rerank(query, candidate_k=20, final_k=5)
        
        # Support both the original list return type and the updated dictionary return type
        top_chunks = retrieval_output["results"] if isinstance(retrieval_output, dict) else retrieval_output
        
        # 2. Extract uniquely ranked papers from chunks
        retrieved_papers = get_unique_papers(top_chunks)
        
        # 3. Calculate metrics
        metrics["Hit Rate@5"].append(hit_rate_at_k(retrieved_papers, relevant_papers, k=5))
        metrics["NDCG@5"].append(ndcg_at_k(retrieved_papers, relevant_papers, k=5))
        metrics["MRR"].append(reciprocal_rank(retrieved_papers, relevant_papers))
        metrics["Precision@5"].append(precision_at_k(retrieved_papers, relevant_papers, k=5))
        metrics["Recall@5"].append(recall_at_k(retrieved_papers, relevant_papers, k=5))

    # ==========================================
    # Print Aggregated Results
    # ==========================================
    print("=" * 40)
    print("🔬 PIPELINE EVALUATION RESULTS")
    print("=" * 40)
    
    for metric_name, values in metrics.items():
        avg_score = sum(values) / len(values) if values else 0.0
        # Format as percentage for Hit Rate and NDCG for readability, others as float
        if metric_name in ["Hit Rate@5", "NDCG@5"]:
            print(f"{metric_name:<15}: {avg_score * 100:.2f}%")
        else:
            print(f"{metric_name:<15}: {avg_score:.4f}")

if __name__ == "__main__":
    # Update these paths relative to where you execute the script from
    GOLDEN_QUERIES_FILE = "./data/evaluation/gold_queries_actual.json"
    PROCESSED_DATA_DIR = "./data/processed"
    
    run_evaluation(GOLDEN_QUERIES_FILE, PROCESSED_DATA_DIR)