# Локально через llama.cpp
# Установите: pip install llama-cpp-python huggingface-hub

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Download Mistral-7B-Instruct GGUF model (Q4_K_M quantization)
model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)
print(f"Model downloaded to: {model_path}")

llm = Llama(
    model_path=model_path,
    chat_format="mistral-instruct",
    n_ctx=2048,
    n_threads=8,   # можно поставить = числу физических ядер CPU
    verbose=False
)

messages = [{"role": "user", "content": "Recommend a good road movies."}]
resp = llm.create_chat_completion(messages=messages)

print(resp["choices"][0]["message"]["content"])