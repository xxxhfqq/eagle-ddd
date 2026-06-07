import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EAGLE_ROOT = ROOT / "third_party" / "EAGLE"
sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.ea_model import EaModel
from experiments.common import load_config, torch_dtype, sync_cuda, reset_cuda_peak_memory, peak_memory_gb, now_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--prompt", default="Explain speculative decoding in simple words.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = EaModel.from_pretrained(
        base_model_path=cfg["base_model_path"],
        ea_model_path=cfg["ea_model_path"],
        torch_dtype=torch_dtype(cfg.get("dtype", "float16")),
        low_cpu_mem_usage=True,
        device_map=cfg.get("device_map", "auto"),
        total_token=cfg.get("total_token", 60),
        depth=cfg.get("depth", 7),
        top_k=cfg.get("top_k", 10),
        threshold=cfg.get("threshold", 1.0),
        use_eagle3=cfg.get("use_eagle3", False),
    )
    model.eval()

    input_ids = model.tokenizer([args.prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    reset_cuda_peak_memory()
    sync_cuda()
    t0 = now_time()

    output = model.eagenerate(
        input_ids,
        temperature=cfg.get("temperature", 0.0),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        log=True,
    )

    sync_cuda()
    t1 = now_time()

    if isinstance(output, tuple):
        output_ids = output[0]
        new_token = output[1] if len(output) > 1 else None
        steps = output[2] if len(output) > 2 else None
    else:
        output_ids = output
        new_token = None
        steps = None

    text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("========== OUTPUT ==========")
    print(text)
    print("========== METRICS ==========")
    print("new_token:", new_token)
    print("steps:", steps)
    print("time:", t1 - t0)
    if new_token is not None:
        print("tokens/s:", new_token / max(t1 - t0, 1e-6))
    print("peak_memory_gb:", peak_memory_gb())


if __name__ == "__main__":
    main()
