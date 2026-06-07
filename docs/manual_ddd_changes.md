# Manual DDD Changes

This document records the manual modification plan for integrating DDD into a clean `SafeAILab/EAGLE` checkout.

The project no longer uses `patches/apply_ddd_patch.py` as the normal workflow, because automatic string replacement is fragile across EAGLE versions. In particular, replacing `total_tokens` globally can corrupt `cnets.py` into invalid Python such as `top_indices.shape[-1]=63`.

## Goal

DDD means **Dynamic Depth Decoding**. It keeps the original EAGLE verifier unchanged and only changes the drafter tree expansion stage.

The intended comparison remains:

1. AR baseline
2. Original EAGLE baseline
3. EAGLE + DDD

## Files to modify manually

Modify these files inside `third_party/EAGLE` only after cloning a clean EAGLE repository:

```text
third_party/EAGLE/eagle/model/ea_model.py
third_party/EAGLE/eagle/model/cnets.py
third_party/EAGLE/eagle/model/cnets1.py
```

## 1. Keep Qwen3 import optional when running Qwen2

Some newer EAGLE versions import Qwen3 code at module import time. If the local `transformers` version does not provide `use_kernel_forward_from_hub`, this can break Qwen2 runs even though Qwen3 is not used.

In `third_party/EAGLE/eagle/model/ea_model.py`, change the top-level Qwen3 import from:

```python
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
```

to:

```python
try:
    from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
except ImportError:
    KVQwen3ForCausalLM = None
```

This keeps Qwen2 loading independent from Qwen3-only dependencies.

## 2. Add DDD parameters to `EaModel`

In `ea_model.py`, add these attributes in `EaModel.__init__` after the model stores its `use_eagle3` or similar configuration fields:

```python
self.enable_ddd = kwargs.get("enable_ddd", False)
self.ddd_max_depth = kwargs.get("ddd_max_depth", None)
self.ddd_check_depths = kwargs.get("ddd_check_depths", (5, 7, 9))
self.ddd_threshold = kwargs.get("ddd_threshold", -2.0)
```

After `self.ea_layer` is created and before or around the line that moves it to the target dtype/device, propagate the settings:

```python
if hasattr(self, "ea_layer"):
    self.ea_layer.enable_ddd = getattr(self, "enable_ddd", False)
    self.ea_layer.ddd_max_depth = getattr(self, "ddd_max_depth", None)
    self.ea_layer.ddd_check_depths = set(getattr(self, "ddd_check_depths", (5, 7, 9)))
    self.ea_layer.ddd_threshold = getattr(self, "ddd_threshold", -2.0)
    self.ea_layer.ddd_depth_history = []
```

## 3. Add DDD state to `cnets.py` and `cnets1.py`

In the drafter `Model.__init__`, keep the original `total_tokens` parameter. Do **not** rename it.

The function signature should look like this, or equivalent:

```python
def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
```

After the original threshold/depth fields are initialized, add:

```python
self.enable_ddd = False
self.ddd_max_depth = None
self.ddd_check_depths = {5, 7, 9}
self.ddd_threshold = -2.0
self.ddd_depth_history = []
self.ddd_last_depth = 0
```

The original EAGLE logic should still keep:

```python
self.total_tokens = total_tokens - 1
```

or the equivalent field used by that EAGLE version.

## 4. Change the draft expansion depth

In `topK_genrate`, find the original line:

```python
depth = self.depth
```

Change it to:

```python
depth = self.ddd_max_depth if getattr(self, "enable_ddd", False) and self.ddd_max_depth is not None else self.depth
self.ddd_last_depth = 0
```

This allows DDD to expand up to a larger maximum depth, while normal EAGLE still uses the original fixed depth.

## 5. Add confidence-based early stop

Inside the draft tree expansion loop, after the cumulative beam scores are updated, add:

```python
self.ddd_last_depth = i + 1
if getattr(self, "enable_ddd", False) and self.ddd_last_depth in getattr(self, "ddd_check_depths", {5, 7, 9}):
    beam_conf = torch.logsumexp(scores.reshape(-1), dim=0)
    if beam_conf.item() < getattr(self, "ddd_threshold", -2.0):
        break
```

The exact insertion point depends on the EAGLE version. It must be after `scores` reflects the current beam cumulative log probabilities.

## 6. Avoid selecting more tokens than were generated

If early stop is enabled, the actual number of generated candidates may be smaller than the original `total_tokens`.

Where the original code selects top draft candidates, prefer this pattern:

```python
select_tokens = min(total_tokens, scores_list.shape[-1])
top_scores, top_indices = torch.topk(scores_list, select_tokens, dim=-1)
```

Then use the actual selected number for later tree-mask and retrieval loops:

```python
selected_tokens = top_indices.shape[-1]
```

Use `selected_tokens` in later loops and tensor sizes that depend on the actual number of selected nodes:

```python
tree_mask = torch.eye(selected_tokens + 1).bool()
for i in range(selected_tokens):
    ...
leaf_num = selected_tokens - noleaf_num
for i in range(selected_tokens + 1):
    ...
maxitem = selected_tokens + 5
```

Important: do not replace every occurrence of `total_tokens` globally. The constructor argument and class field should stay as `total_tokens`.

## 7. Record actual DDD depth

Before returning from `topK_genrate`, add:

```python
if getattr(self, "enable_ddd", False):
    self.ddd_depth_history.append(getattr(self, "ddd_last_depth", depth))
```

This allows the benchmark script to summarize actual dynamic depths.

## 8. Validation commands

After manual edits, run:

```bash
python -m py_compile third_party/EAGLE/eagle/model/ea_model.py
python -m py_compile third_party/EAGLE/eagle/model/cnets.py
python -m py_compile third_party/EAGLE/eagle/model/cnets1.py

python - <<'PY'
from eagle.model.ea_model import EaModel
print("EaModel import ok")
PY

bash scripts/03_run_smoke_test.sh
```

If smoke test fails, inspect the traceback first. Do not rerun the automatic patch script.
