# Experiment Plan

## Goal

实现并评估 DDD（Dynamic Depth Decoding，动态深度解码）对 EAGLE 推理速度的影响。

核心比较对象：

1. AR decoding：普通自回归生成。
2. EAGLE baseline：官方 EAGLE 推理。
3. EAGLE + DDD：加入动态深度早停的 EAGLE 推理。

## Main idea

原始 EAGLE 会按照固定深度扩展 draft tree。DDD 的思想是：

- 最多扩展到更深的 `ddd_max_depth`。
- 在指定层数 `ddd_check_depths = [5, 7, 9]` 检查 beam 的整体置信度。
- 如果 `logsumexp(scores)` 低于 `ddd_threshold`，就提前停止扩展。

这样可以避免在低置信度 prefix 上继续生成大量最终会被 verifier 拒绝的候选 token。

## Implementation strategy

本项目采用轻度侵入式修改：

- 不改 target model。
- 不改 verifier 的接受 / 拒绝逻辑。
- 只改 drafter 的 tree expansion 阶段。
- 通过 `enable_ddd` 开关控制是否启用 DDD。

这样可以保证 baseline 和 DDD 在同一套代码中公平对比。

## Metrics

建议至少记录：

| Metric | Meaning |
|---|---|
| tokens/s | 总吞吐速度 |
| total time | 总耗时 |
| avg accepted length | 平均每轮 target verification 接受多少 token |
| steps | verifier 调用轮数 |
| peak memory | 峰值显存 |
| actual depth distribution | DDD 每轮实际扩展深度分布 |

## Ablation

DDD 阈值建议测试：

```text
-3.0, -2.0, -1.0, 0.0
```

预期现象：

- 阈值太低：几乎不提前停，接近原始 EAGLE。
- 阈值太高：停得过早，accepted length 下降，可能变慢。
- 中间阈值：draft 计算减少，同时 accepted length 下降不明显，可能加速。

## Minimum successful result

最低完成标准：

1. 能跑通 AR、EAGLE、EAGLE+DDD。
2. 能生成 `results/summary.csv`。
3. 能画出 throughput 对比图。
4. 能说明 DDD 的 depth distribution 不是固定的，而是动态变化的。
