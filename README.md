# EAGLE DDD Final Project

这个仓库用于完成 USTC ML System final project 中的 EAGLE / EAGLE-2 推理优化实验。

核心目标：在官方 `SafeAILab/EAGLE` 的基础上，加入 **DDD（Dynamic Depth Decoding，动态深度解码）**，比较：

1. AR（Autoregressive，自回归推理）
2. 原始 EAGLE 推理
3. EAGLE + DDD 推理

本仓库不保存大模型权重，只保存实验脚手架、补丁脚本、benchmark 脚本和报告材料。

---

## 目录结构

```text
.
├── environment.yml                  # Conda 环境
├── requirements-extra.txt           # 额外 Python 依赖
├── scripts/                         # 一键脚本
├── patches/                         # 自动修改 EAGLE 源码的补丁脚本
├── experiments/                     # smoke test、benchmark、统计、画图
├── configs/                         # 模型路径配置模板
├── docs/                            # 实验计划和报告大纲
└── results/                         # 运行结果，默认不提交大文件
```

---

## 0. 基本概念

- **EAGLE**: Extrapolation Algorithm for Greater Language-model Efficiency，用小的 drafter 预测未来 token，再由 target model 一次性验证。
- **DDD**: Dynamic Depth Decoding，动态深度解码。它不固定扩展 draft tree 的深度，而是在若干层检查 beam 的整体置信度，低于阈值就提前停止。
- **AR**: Autoregressive，自回归推理，一个 token 一个 token 地生成。
- **GPU**: Graphics Processing Unit，图形处理器。
- **CUDA**: Compute Unified Device Architecture，NVIDIA 的 GPU 并行计算平台。

---

## 1. 克隆仓库

```bash
git clone https://github.com/xxxhfqq/eagle-ddd.git
cd eagle-ddd
```

---

## 2. 下载官方 EAGLE 代码

```bash
bash scripts/00_clone_eagle.sh
```

这会把官方 EAGLE 仓库放到：

```text
third_party/EAGLE
```

---

## 3. 创建环境

```bash
bash scripts/01_setup_env.sh
conda activate eagle_ddd
```

如果服务器 PyTorch / CUDA 版本冲突，优先按照服务器已有 CUDA 版本单独安装 PyTorch，然后再执行：

```bash
pip install -e third_party/EAGLE
pip install -r requirements-extra.txt
```

---

## 4. 应用 DDD 补丁

```bash
bash scripts/02_apply_ddd_patch.sh
```

这个脚本会自动修改：

```text
third_party/EAGLE/eagle/model/ea_model.py
third_party/EAGLE/eagle/model/cnets.py
third_party/EAGLE/eagle/model/cnets1.py
```

它会加入：

```text
enable_ddd
ddd_max_depth
ddd_check_depths
ddd_threshold
ddd_depth_history
```

注意：补丁脚本会保留 `.bak` 备份文件。

---

## 5. 设置模型路径

先复制配置模板：

```bash
cp configs/vicuna7b_eagle.yaml configs/local.yaml
```

然后编辑：

```bash
vim configs/local.yaml
```

把下面两个路径改成你服务器上的真实路径：

```yaml
base_model_path: /workspace/models/vicuna-7b-v1.3
ea_model_path: /workspace/models/EAGLE-Vicuna-7B-v1.3
```

如果你用 Qwen2，可以改用：

```bash
cp configs/qwen2_7b_eagle.yaml configs/local.yaml
```

---

## 6. 跑 smoke test

先只生成一个样例，确认模型能加载、能生成：

```bash
bash scripts/03_run_smoke_test.sh
```

如果这一步跑不通，先不要做正式 benchmark。

---

## 7. 跑正式实验

```bash
bash scripts/04_run_benchmark.sh
```

会依次运行：

```text
ar
eagle
eagle_ddd, threshold=-3.0
eagle_ddd, threshold=-2.0
eagle_ddd, threshold=-1.0
eagle_ddd, threshold=0.0
```

结果写入：

```text
results/*.jsonl
```

---

## 8. 汇总结果和画图

```bash
bash scripts/05_collect_results.sh
```

输出：

```text
results/summary.csv
results/fig_tokens_per_second.png
results/fig_avg_accept_len.png
results/fig_ddd_depth_hist.png
```

---

## 9. 推荐报告主线

报告里不要只说“我加了 DDD”，而要按下面逻辑写：

1. 先 profile 原始 EAGLE，说明固定深度 draft tree 有浪费。
2. 提出 DDD：根据 beam log probability 的整体置信度动态停止扩展。
3. 强调 DDD 只改变 proposal，不改变 verifier，因此不会破坏 speculative decoding 的基本正确性。
4. 做 AR、EAGLE、EAGLE+DDD 对比。
5. 做 threshold ablation，说明阈值过大/过小都会影响速度。

详细写法见：

```text
docs/report_outline.md
```
