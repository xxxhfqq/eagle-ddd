import json
import time
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def load_prompts(path=None):
    if path is None:
        return [
            "Explain speculative decoding in simple words.",
            "Write a short introduction to GPU memory hierarchy.",
            "Why can draft models accelerate large language model inference?",
            "Give three practical tips for optimizing PyTorch inference.",
            "Summarize the difference between latency and throughput.",
        ]
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now_time():
    return time.time()
