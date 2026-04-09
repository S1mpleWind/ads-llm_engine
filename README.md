# Project：从 Qwen3.5 到 KV Cache

本仓库通过在真实的 **Qwen3.5** 架构上手动实现推理引擎，协助同学系统理解**大语言模型推理**的核心机制——尤其是 **KV Cache** 这一推理优化中极其重要的数据结构。

## 目录

- [项目总体情况](#项目总体情况)
- [环境配置与运行](#环境配置与运行)
- [Qwen3.5-0.8B 架构要点](#qwen35-0.8b-架构要点)
- [文档导航](#文档导航)
- [许可与声明](#许可与声明)

---

## 项目总体情况

项目分为两个阶段，每个阶段都有独立的说明文档：

| 阶段 | 内容 | 文档 |
|------|------|------|
| **Phase 1** | 配置环境、拉取权重，跑通统一基线推理命令 | 本文档 |
| **Phase 2** | 在已有推理框架中补全 KV Cache 逻辑，观察 decode 速度提升 | [phase2.md](phase2.md) |


---

## 环境配置与运行

**依赖要求**：Python ≥ 3.11，使用 `uv` 管理依赖。

```bash
# 安装依赖并激活环境
uv sync && source .venv/bin/activate

# Phase 1：运行基线推理（首次运行会下载 Qwen3.5-0.8B 权重）
chmod +x run.sh
./run.sh
```

> 如果下载极慢或卡住，请检查网络环境（默认使用 `hf-mirror.com` 镜像）。

**自定义参数**：

```bash
python main.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Your prompt here" \
  --stream true \
  --benchmark true \
  --max-new-tokens 256
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-new-tokens` | 最多生成的 token 数 | 4096 |
| `--temperature` | 采样温度 | 0.7 |
| `--top-p` / `--top-k` | 核采样 / top-k 采样 | 0.9 / 40 |
| `--stream` | 流式输出 | false |
| `--benchmark` | 输出 prefill/decode 性能统计 | false |
| `--no-cache` | 禁用 KV Cache（Phase 1 基线模式） | false |

---

## Qwen3.5-0.8B 架构要点

**Qwen3.5** 是通义千问团队开源的最新一代 Decoder-only 因果语言模型，**Qwen/Qwen3.5-0.8B** 是该系列中的稠密小模型（无 MoE），适合本地部署与课堂演示，与更大规格成员共享同一套设计思想。

与常见的「纯 Softmax 自注意力」架构相比，Qwen3.5 有两个关键特点：

**1. 混合注意力机制**（继承 Qwen3-Next 思路）

在层堆叠中，模型以 **3:1** 的比例交替使用两种注意力模块：

- **Full Attention 层**（每 4 层出现 1 次）：标准 softmax 注意力 + RoPE 位置编码，配合 sigmoid 门控；需要随序列增长的 **KV Cache**。
- **Linear Attention 层**（占 75%）：**Gated Delta Net** 结构，以近似线性复杂度处理长上下文；维护**固定大小的递推状态**，不随序列增长。

这种混合设计使 Qwen3.5 在 32K 上下文时比纯 Full Attention 快 **8.6 倍**，256K 时快 **19 倍**，同时推理质量几乎无损。

**2. 原生多模态设计**

Qwen3.5 系列对文本、图像、视频一体化设计。本仓库中 **0.8B** 主要承担文本推理，但其架构与多模态大版本完全一致。

---

## 文档导航

本项目有三个文档，建议按顺序阅读：

---

### 📖 [concepts.md](concepts.md) — 背景知识

**阅读时机**：开始 Phase 2 之前，建立必要的理论基础。

涵盖内容：

| 章节 | 内容摘要 |
|------|---------|
| Self-Attention 计算步骤 | Q/K/V 投影、scaled dot-product、softmax、多头注意力 |
| Decoder-only 与 Causal Mask | 因果性如何使 K/V 可以安全缓存，为什么不缓存 Q |
| Prefill 与 Decode 两阶段 | 两阶段的计算特征，以及负载差异 |
| 有无 KV Cache 的效率差异 | decode 阶段从全序列重算变为只算新 token |
| KV Cache 的产业意义 | 显存、带宽瓶颈，以及各种优化方向 |
| Linear Attention 原理 | 如何用固定状态矩阵替代增长的 KV Cache，优缺点对比 |
| Gated Delta Net（选读） | Qwen3.5 线性注意力层的具体实现与推理路径 |
| 混合缓存对比 | `recurrent_state` / `conv_state` vs KV Cache 的形状、更新方式对比 |
| 延伸阅读 | 论文、博客、课程推荐 |

---

### 🔧 [phase2.md](phase2.md) — Phase 2 实现任务

**阅读时机**：读完 concepts.md 之后，开始动手实现。

涵盖内容：

| 章节 | 内容摘要 |
|------|---------|
| 任务总览 | Phase 1 vs Phase 2 的核心差异，Prefill/Decode 分离的目标 |
| Task 1：动态缓存类 | 实现 `Qwen3_5DynamicCache`（`cache.py`） |
| Task 2：Full Attention 缓存 | 在 `qwen3_5_attention_forward()` 中接入 KV Cache |
| Task 3：Linear Attention 缓存 | 实现单步卷积更新与递推，`qwen3_5_linear_attn_forward()` 分支 |
| Task 4：Forward 传递缓存 | `qwen3_5_text_forward()` 中创建、传递、返回缓存 |
| Task 5：Decode 循环改造 | Prefill 一次 + Decode 逐 token，记录性能统计 |
| 定位 TODO / 运行验证 | `grep` 命令定位所有填空位置，运行 `test_phase2.py` 观察速度提升 |
| 提交要求 | `test_phase2.py` 通过；最终提交内容： `<学号>_project_phase2.zip`，内含实现代码和测试结果截图，详见[phase2.md](phase2.md)  |

---

## 许可与声明

模型权重版权归 **Qwen / Hugging Face** 及相应许可协议所有；本仓库课程代码以课程规定为准。
