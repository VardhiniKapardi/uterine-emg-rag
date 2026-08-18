# src/generator.py
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class LocalQwenGenerator:
    def __init__(self):
        print("Downloading/Loading Quantized Qwen2.5-3B...")
        # Automatically downloads the optimized GGUF file to your local cache
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        
        # Initialize the CPU-optimized engine
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,      # Context window size
            n_threads=4,     # Adjust based on your CPU cores
            verbose=False    # Suppresses C++ backend logs
        )

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a scientific research assistant specializing in uterine electromyography (EMG/EHG).
Answer the user's question using ONLY the information provided in the context below.

Context:
{context}

Question:
{query}

Answer:"""

        # Generate response using the local CPU
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful biomedical research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        
        return response["choices"][0]["message"]["content"]