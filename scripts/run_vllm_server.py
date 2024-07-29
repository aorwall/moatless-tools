import os
import argparse
import subprocess
import os
from huggingface_hub import login
from transformers import AutoModel
import socket


"""
models

"NousResearch/Meta-Llama-3-8B-Instruct"
"meta-llama/Meta-Llama-3-70B"

"""

# from transformers import AutoModel



os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ["HF_TOKEN"] = "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"

# login(token=os.environ.get("HF_TOKEN"))  # Or directly pass your token as a string

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
DEEPSEEK_MODELS = ["deepseek/deepseek-coder", "deepseek/deepseek-chat"]

MODELS = {
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "8002",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "8000",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "8000",
    "Qwen/Qwen2-7B-Instruct": "8003",
    "Qwen/Qwen2-57B-A14B-Instruct": "8000",
    "Qwen/Qwen2-72B-Instruct": "8000",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "8006",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct": "8007",
    "deepseek-ai/deepseek-coder-33b-instruct": "8008",
}

def find_available_port(start_port=8000, max_port=9000):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise IOError("No free ports")

def run_vllm_server(model_key, num_gpus):
    model_name = model_key
    api_key = "token-abc123"  # Replace with your actual API key
    port = find_available_port()
    command = f"python -m vllm.entrypoints.openai.api_server \
    --model {model_name} \
    --dtype auto \
    --api-key {api_key} \
    --tensor-parallel-size {num_gpus} \
    --port {port}"
    process = subprocess.Popen(command, shell=True)
    return process, port

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple vLLM servers with specified models and GPUs")
    parser.add_argument("--config", nargs="+", help="Configurations in the format: model_key:num_gpus")
    args = parser.parse_args()

    processes = []
    for config in args.config:
        model_key, num_gpus = config.split(":")
        print(f"Starting server for model {model_key} with {num_gpus} GPUs")
        process, port = run_vllm_server(model_key, int(num_gpus))
        print(f"Server started on port {port}")
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()