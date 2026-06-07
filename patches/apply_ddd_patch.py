import argparse
from pathlib import Path


HEADER = "# Auto-patched by eagle-ddd project"


def read_text(path):
    return path.read_text(encoding="utf-8")


def write_text(path, text):
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_ea_model(path):
    text = read_text(path)
    if "enable_ddd" in text:
        print(f"[skip] {path} already contains enable_ddd")
        return

    marker = "        self.use_eagle3 = use_eagle3\n"
    insert = marker + "        self.enable_ddd = kwargs.get('enable_ddd', False)\n        self.ddd_max_depth = kwargs.get('ddd_max_depth', None)\n        self.ddd_check_depths = kwargs.get('ddd_check_depths', (5, 7, 9))\n        self.ddd_threshold = kwargs.get('ddd_threshold', -2.0)\n"
    if marker in text:
        text = text.replace(marker, insert, 1)
    else:
        print(f"[warn] cannot find EaModel init marker in {path}; only patching ea_layer if possible")

    marker2 = "        self.ea_layer = self.ea_layer.to(self.base_model.dtype).to(self.base_model.device)\n"
    insert2 = "        if hasattr(self, 'ea_layer'):\n            self.ea_layer.enable_ddd = getattr(self, 'enable_ddd', False)\n            self.ea_layer.ddd_max_depth = getattr(self, 'ddd_max_depth', None)\n            self.ea_layer.ddd_check_depths = set(getattr(self, 'ddd_check_depths', (5, 7, 9)))\n            self.ea_layer.ddd_threshold = getattr(self, 'ddd_threshold', -2.0)\n            self.ea_layer.ddd_depth_history = []\n" + marker2
    if marker2 in text:
        text = text.replace(marker2, insert2, 1)
    else:
        print(f"[warn] cannot find ea_layer.to marker in {path}")

    write_text(path, text)
    print(f"[ok] patched {path}")


def patch_cnets_file(path):
    text = read_text(path)
    if "ddd_depth_history" in text:
        print(f"[skip] {path} already patched")
        return

    text = text.replace(
        "        self.threshold = threshold\n",
        "        self.threshold = threshold\n        self.enable_ddd = False\n        self.ddd_max_depth = None\n        self.ddd_check_depths = {5, 7, 9}\n        self.ddd_threshold = -2.0\n        self.ddd_depth_history = []\n",
        1,
    )

    # EAGLE code normally uses depth = self.depth inside topK_genrate.
    text = text.replace(
        "        depth = self.depth\n",
        "        depth = self.ddd_max_depth if getattr(self, 'enable_ddd', False) and self.ddd_max_depth is not None else self.depth\n        self.ddd_last_depth = 0\n",
        1,
    )

    # Insert DDD check after scores_list is updated. Different EAGLE versions may have the exact line below.
    target = "            scores_list = torch.cat((scores_list, scores), dim=1)\n"
    insert = target + "            self.ddd_last_depth = i + 1\n            if getattr(self, 'enable_ddd', False) and self.ddd_last_depth in getattr(self, 'ddd_check_depths', {5, 7, 9}):\n                beam_conf = torch.logsumexp(scores.reshape(-1), dim=0)\n                if beam_conf.item() < getattr(self, 'ddd_threshold', -2.0):\n                    break\n"
    if target in text:
        text = text.replace(target, insert, 1)
    else:
        print(f"[warn] cannot find scores_list update marker in {path}; DDD early stop may not be inserted")

    # Avoid topk larger than available candidates after early stop.
    text = text.replace(
        "top_scores, top_indices = torch.topk(scores_list, total_tokens, dim=-1)",
        "select_tokens = min(total_tokens, scores_list.shape[-1])\n        top_scores, top_indices = torch.topk(scores_list, select_tokens, dim=-1)",
        1,
    )
    text = text.replace("total_tokens + 1", "top_indices.shape[-1] + 1")
    text = text.replace("total_tokens", "top_indices.shape[-1]")

    # Record actual DDD depth before returning.
    text = text.replace(
        "        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids\n",
        "        if getattr(self, 'enable_ddd', False):\n            self.ddd_depth_history.append(getattr(self, 'ddd_last_depth', depth))\n        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids\n",
        1,
    )

    write_text(path, text)
    print(f"[ok] patched {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eagle-root", default="third_party/EAGLE")
    args = parser.parse_args()

    root = Path(args.eagle_root)
    if not root.exists():
        raise SystemExit(f"EAGLE root not found: {root}")

    ea_model = root / "eagle" / "model" / "ea_model.py"
    cnets = root / "eagle" / "model" / "cnets.py"
    cnets1 = root / "eagle" / "model" / "cnets1.py"

    if ea_model.exists():
        patch_ea_model(ea_model)
    else:
        print(f"[warn] missing {ea_model}")

    for p in [cnets, cnets1]:
        if p.exists():
            patch_cnets_file(p)
        else:
            print(f"[warn] missing {p}")

    print("[done] DDD patch attempted. Please run smoke test next.")


if __name__ == "__main__":
    main()
