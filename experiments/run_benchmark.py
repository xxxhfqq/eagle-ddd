import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EAGLE_ROOT = ROOT / "third_party" / "EAGLE"
sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.ea_model import EaModel
from experiments.common import (
    append_jsonl,
    load_config,
    load_prompts,
    now_time,
    peak_memory_gb,
    reset_cuda_peak_memory,
    sync_cuda,
    torch_dtype,
)


def load_eagle_model(cfg, enable_ddd=False, ddd_threshold=-2.0):
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
        enable_ddd=enable_ddd,
        ddd_max_depth=cfg.get("ddd_max_depth", 11),
        ddd_check_depths=tuple(cfg.get("ddd_check_depths", [5, 7, 9])),
        ddd_threshold=ddd_threshold,
    )
    model.eval()
    return model


def run_eagle(cfg, prompts, output_path, enable_ddd=False, ddd_threshold=-2.0):
    model = load_eagle_model(cfg, enable_ddd=enable_ddd, ddd_threshold=ddd_threshold)
    tokenizer = model.tokenizer

    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer([prompt]).input_ids
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
            new_token = output_ids.shape[-1] - input_ids.shape[-1]
            steps = None

        actual_depths = []
        if enable_ddd and hasattr(model, "ea_layer") and hasattr(model.ea_layer, "ddd_depth_history"):
            actual_depths = list(model.ea_layer.ddd_depth_history)
            model.ea_layer.ddd_depth_history.clear()

        record = {
            "idx": idx,
            "method": "eagle_ddd" if enable_ddd else "eagle",
            "ddd_threshold": ddd_threshold if enable_ddd else None,
            "prompt": prompt,
            "output": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "new_tokens": int(new_token) if new_token is not None else None,
            "steps": int(steps) if steps is not None else None,
            "time": t1 - t0,
            "tokens_per_second": float(new_token / max(t1 - t0, 1e-6)) if new_token is not None else None,
            "avg_accept_len": float(new_token / max(steps, 1)) if new_token is not None and steps is not None else None,
            "peak_memory_gb": peak_memory_gb(),
            "ddd_actual_depths": actual_depths,
        }
        append_jsonl(output_path, record)
        print(record)


def run_ar(cfg, prompts, output_path):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_path"],
        torch_dtype=torch_dtype(cfg.get("dtype", "float16")),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()

    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[-1]

        reset_cuda_peak_memory()
        sync_cuda()
        t0 = now_time()
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=cfg.get("max_new_tokens", 128),
        )
        sync_cuda()
        t1 = now_time()

        new_token = output_ids.shape[-1] - input_len
        record = {
            "idx": idx,
            "method": "ar",
            "prompt": prompt,
            "output": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "new_tokens": int(new_token),
            "steps": int(new_token),
            "time": t1 - t0,
            "tokens_per_second": float(new_token / max(t1 - t0, 1e-6)),
            "avg_accept_len": 1.0,
            "peak_memory_gb": peak_memory_gb(),
            "ddd_actual_depths": [],
        }
        append_jsonl(output_path, record)
        print(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--method", choices=["ar", "eagle", "eagle_ddd"], required=True)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ddd-threshold", type=float, default=-2.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = load_prompts(args.prompts)

    if args.method == "ar":
        run_ar(cfg, prompts, args.output)
    elif args.method == "eagle":
        run_eagle(cfg, prompts, args.output, enable_ddd=False)
    else:
        run_eagle(cfg, prompts, args.output, enable_ddd=True, ddd_threshold=args.ddd_threshold)


if __name__ == "__main__":
    main()
