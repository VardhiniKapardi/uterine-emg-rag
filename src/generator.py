# src/generator.py
import time
import psutil
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class LocalQwenGenerator:
    def __init__(self):
        print("Downloading/Loading Quantized Qwen2.5-3B...")
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,      
            n_threads=4,     
            verbose=False    
        )

    def generate_answer(self, query: str, context: str) -> dict:
        prompt = f"""You are a scientific research assistant specializing in uterine electromyography (EMG/EHG).
Answer the user's question using ONLY the information provided in the context below.

Context:
{context}

Question:
{query}

Answer:"""

        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.perf_counter()
        
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful biomedical research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        
        end_time = time.perf_counter()
        ram_after = process.memory_info().rss / (1024 * 1024)
        
        generation_time = end_time - start_time
        completion_tokens = response["usage"]["completion_tokens"]
        tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
        
        return {
            "content": response["choices"][0]["message"]["content"],
            "metrics": {
                "generation_time_sec": round(generation_time, 2),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "total_tokens": response["usage"]["total_tokens"],
                "approx_ram_mb": round(ram_after, 2),
                "ram_delta_mb": round(ram_after - ram_before, 2)
            }
        }