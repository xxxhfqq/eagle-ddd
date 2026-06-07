# Report Outline

## 1. Introduction

LLM inference is expensive because autoregressive decoding generates one token per target model forward pass. Speculative decoding accelerates inference by using a lightweight drafter to propose candidate tokens and using the original target model to verify them.

EAGLE is an efficient speculative decoding method. However, its draft tree expansion can still waste computation when the drafter is uncertain. In such cases, continuing to expand the tree may generate many low-quality candidates that are later rejected by the target model.

This project proposes a training-free optimization: Dynamic Depth Decoding (DDD). DDD adaptively stops draft tree expansion based on the confidence of current beam candidates.

## 2. Background

### Autoregressive decoding

Autoregressive decoding generates tokens one by one. It is simple but slow because each generated token requires a target model forward pass.

### Speculative decoding

Speculative decoding uses a small drafter to propose multiple tokens. The target model verifies these tokens in parallel. If more draft tokens are accepted, fewer target model calls are needed.

### EAGLE

EAGLE predicts future feature-level information to construct draft candidates efficiently. It uses a drafter and a target model verifier.

## 3. Motivation

Fixed-depth draft tree expansion is not always efficient.

For easy prompts, the drafter may be confident and can safely expand deeper. For hard prompts, the drafter may be uncertain, and expanding deeper creates low-quality candidates. These candidates consume draft time and verification budget but contribute little to accepted length.

Therefore, draft depth should be input-adaptive.

## 4. Method

DDD introduces a confidence-based early stop mechanism into draft tree expansion.

At selected depths, such as 5, 7, and 9, DDD computes the beam-level confidence:

```text
H = logsumexp(path_log_probs)
```

If `H < threshold`, the draft expansion stops early.

Important design choice:

- DDD only changes the proposal stage.
- DDD does not change the target model.
- DDD does not change the verifier acceptance/rejection rule.

This makes the modification safer and easier to evaluate.

## 5. Implementation

The implementation is based on `SafeAILab/EAGLE`.

Modified components:

```text
eagle/model/ea_model.py
eagle/model/cnets.py
eagle/model/cnets1.py
```

Added parameters:

```text
enable_ddd
ddd_max_depth
ddd_check_depths
ddd_threshold
```

The benchmark scripts compare AR, EAGLE, and EAGLE+DDD using the same prompts and model paths.

## 6. Experiments

### Setup

Report:

- GPU type
- CUDA version
- PyTorch version
- base model
- EAGLE model
- max new tokens
- prompts or dataset

### Baselines

- AR decoding
- Original EAGLE
- EAGLE + DDD

### Metrics

- tokens/s
- total time
- average accepted length
- verifier steps
- peak GPU memory
- actual draft depth distribution

## 7. Results

Recommended tables:

| Method | tokens/s | Avg accepted length | Total time | Peak memory |
|---|---:|---:|---:|---:|
| AR | | | | |
| EAGLE | | | | |
| EAGLE + DDD | | | | |

Recommended figures:

1. Throughput comparison.
2. Average accepted length comparison.
3. DDD actual depth distribution.

## 8. Analysis

Discuss threshold effects:

- Too low: DDD rarely stops early, close to baseline.
- Too high: DDD stops too aggressively, accepted length decreases.
- Proper threshold: reduces draft overhead while preserving accepted length.

Discuss why DDD may or may not speed up:

- If drafter time is a significant bottleneck, DDD helps.
- If verifier dominates the runtime, DDD may have limited speedup.
- If early stopping is too aggressive, target calls may increase.

## 9. Limitations

- Threshold depends on model and prompt distribution.
- The current implementation is a course-project version and may need further tuning for production serving.
- Batch inference behavior may differ from single-prompt inference.

## 10. Conclusion

DDD is a simple, training-free optimization for EAGLE-style speculative decoding. It improves draft efficiency by adapting the tree depth to drafter confidence. The method is easy to implement, does not require retraining, and is suitable for course-level system optimization experiments.
