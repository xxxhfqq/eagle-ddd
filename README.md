# EAGLE DDD Final Project

这个仓库用于完成 USTC ML System final project 中的 EAGLE / EAGLE-2 推理优化实验。

核心目标：在官方 `SafeAILab/EAGLE` 的基础上，保留 **DDD（Dynamic Depth Decoding，动态深度解码）** 的实验思路，比较：

1. AR（Autoregressive，自回归推理）
2. 原始 EAGLE 推理
3. EAGLE + DDD 推理

本仓库不保存大模型权重，只保存实验脚手架、benchmark 脚本、报告材料和手动修改说明。

---

## 目录结构

```text
.
├── environment.yml                  # Conda 环境模板
├── requirements-extra.txt           # 额外 Python 依赖
├── scripts/                         # 一键脚本
├── patches/                         # 旧版自动补丁脚本，仅保留作参考，不作为默认流程
├── experiments/                     # smoke test、benchmark、统计、画图
├── configs/                         # 模型路径配置模板
├── docs/                            # 实验计划、报告大纲、手动 DDD 修改说明
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

大陆网络建议使用公开库加速地址：

```bash
git clone https://gh-proxy.com/https://github.com/xxxhfqq/eagle-ddd.git
cd eagle-ddd
```

如果你的网络可以直接访问 GitHub，也可以使用原地址。

---

## 2. 下载官方 EAGLE 代码

```bash
bash scripts/00_clone_eagle.sh
```

脚本默认使用公开库加速地址：

```text
https://gh-proxy.com/https://github.com/SafeAILab/EAGLE.git
```

这会把官方 EAGLE 仓库放到：

```text
third_party/EAGLE
```

如果你要临时切换地址，可以覆盖 `EAGLE_REPO_URL`：

```bash
EAGLE_REPO_URL=https://gh-proxy.com/https://github.com/SafeAILab/EAGLE.git bash scripts/00_clone_eagle.sh
```

---

## 3. 创建独立环境

不要在 `base` 环境里安装。建议新建一个专门环境：

```bash
conda create -n eagle_ddd python=3.10 -y
conda activate eagle_ddd
```

如果你已经在 `base` 里失败过，问题不大，后面只要切到 `eagle_ddd` 环境重新装即可。

---

## 4. 先安装 PyTorch

EAGLE 官方安装依赖里可能写死 `torch==2.0.1`，但服务器 CUDA 和国内 PyPI 镜像不一定匹配。因此本项目推荐：**先手动安装 PyTorch，再用 `--no-deps` 安装 EAGLE 本体**。

如果服务器是 CUDA 12.1 附近，可以先试：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你的服务器已经有合适的 PyTorch，也可以先检查：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda:', torch.version.cuda)
PY
```

只要 `cuda available: True`，就可以继续下一步。

---

## 5. 安装项目依赖和 EAGLE 本体

```bash
bash scripts/01_setup_env.sh
```

这个脚本会执行类似：

```bash
python -m pip install numpy pandas matplotlib pyyaml tqdm sentencepiece protobuf accelerate transformers safetensors huggingface-hub
python -m pip install --no-deps -e third_party/EAGLE
```

其中 `--no-deps` 表示 **no dependencies，不安装依赖**。这样可以避免 EAGLE 强行安装固定版本的 `torch==2.0.1`，也避免污染当前环境。

---

## 6. 手动加入 DDD 修改

默认流程已经**不再使用自动补丁脚本**。

不要运行：

```bash
bash scripts/02_apply_ddd_patch.sh
```

这个脚本现在只会打印提示，不会修改源码。旧的 `patches/apply_ddd_patch.py` 仅保留作参考，不作为默认流程。

新的推荐方式是：在干净的 `third_party/EAGLE` 上，按照下面文档手动修改：

```text
docs/manual_ddd_changes.md
```

手动修改的核心思想仍然是：

```text
1. 不改 target model。
2. 不改 verifier 的接受 / 拒绝逻辑。
3. 只在 drafter 的 tree expansion 阶段加入动态深度早停。
4. 通过 enable_ddd 开关控制是否启用 DDD。
```

这样可以保留 DDD 的实验思路，同时避免字符串补丁误伤源码。

---

## 7. 设置模型路径

先复制配置模板：

```bash
cp configs/qwen2_7b_eagle.yaml configs/local.yaml
```

然后编辑：

```bash
vim configs/local.yaml
```

如果模型放在当前项目目录，可以写成：

```yaml
base_model_path: ./models/Qwen2-7B-Instruct
ea_model_path: ./models/EAGLE-Qwen2-7B-Instruct
```

如果模型放在其他目录，就把路径改成真实路径。

---

## 8. 跑 smoke test

先只生成一个样例，确认模型能加载、能生成：

```bash
bash scripts/03_run_smoke_test.sh
```

如果这一步跑不通，先不要做正式 benchmark。

---

## 9. 跑正式实验

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

注意：`eagle_ddd` 需要你已经按 `docs/manual_ddd_changes.md` 手动加入 DDD 代码。否则只跑 AR 和原始 EAGLE。

---

## 10. 汇总结果和画图

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

## 11. 推荐报告主线

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
