# main.py
from src.retriever import RAGRetriever
from src.generator import LocalQwenGenerator

def main():
    # 1. Initialize components
    processed_data_path = "./data/processed"
    
    retriever = RAGRetriever(processed_dir=processed_data_path)
    generator = LocalQwenGenerator()
    
    # 2. Define Query
    query = "What are the characteristics of uterine EMG signals?"
    print(f"\nQuery: {query}")
    
    # 3. Retrieve Context
    top_chunks = retriever.retrieve_and_rerank(query)
    
    context = "\n\n".join([
        f"[Source: {chunk['paper']}] {chunk['text']}" 
        for chunk in top_chunks
    ])
    
    # 4. Generate Answer natively on CPU
    print("\nGenerating answer on CPU...")
    answer = generator.generate_answer(query, context)
    
    print("\n=== ANSWER ===")
    print(answer)

if __name__ == "__main__":
    main()