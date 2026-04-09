# Phase 2：理解并实现 KV Cache

> **前置阅读**：开始本文档之前，请先阅读 [concepts.md](concepts.md)，其中涵盖 Self-Attention、Decoder-only、KV Cache、Linear Attention 与混合缓存的背景知识，有助于各位完成 phase2。

## 目录

- [任务总览](#任务总览)
- [任务与实现要求](#任务与实现要求)
  - [Task 1：动态缓存类](#task-1动态缓存类-tiny_inferencecachepy)
  - [Task 2：Full Attention 层的 KV Cache](#task-2full-attention-层的-kv-cache-tiny_inferencemanual_attentionpy)
  - [Task 3：Linear Attention 层的缓存](#task-3linear-attention-层的缓存-tiny_inferencemanual_linearpy)
  - [Task 4：Forward 中创建缓存](#task-4forward-中创建缓存-tiny_inferencemanual_qwen3_5py)
  - [Task 5：Prefill + Decode 解码循环](#task-5prefill--decode-解码循环-tiny_inferencemanual_decodingpy)
- [如何定位需要写的代码](#如何定位需要写的代码)
- [如何运行与验证](#如何运行与验证)
- [提交要求](#提交要求)

---

## 任务总览

Phase 1 中，推理引擎在每个 decode 步都将**从 prompt 到当前所有已生成 token 的完整序列**重新喂入模型，等价于每生成一个 token 就重做一次完整 prefill。这导致 decode 阶段的耗时**随序列增长呈二次增长**。

Phase 2 的目标：**实现 KV Cache**，将推理分为两阶段：

| 阶段 | 做了什么 | 输入 |
|------|---------|------|
| **Prefill** | 一次性处理完整 prompt，建立所有层的缓存 | 完整 prompt tokens |
| **Decode** | 每步只输入**刚生成的 1 个 token**，从缓存中读取历史信息 | 单个 token + 缓存 |

你要写的代码并不多，希望你能够在这个过程中能够对混合注意力和KV Cache有更多了解。
完成后，你应当能观察到 **decode 阶段速度显著提升**。

---

## 任务与实现要求

所有需要你实现的代码位置都用以下标记包裹：

```python
# ===== TODO: KV Cache - <描述> (START) =====
...（需要实现的代码）...
# ===== TODO: KV Cache - <描述> (END) =====
```

---

### Task 1：动态缓存类 (`tiny_inference/cache.py`)

这是整个 KV Cache 的核心数据结构。它负责为模型的每一层维护运行时状态，让 decode 阶段能从"上次停下的地方"继续，而不必重新计算整段历史序列。

由于模型是混合架构，不同类型的层需要不同的缓存方式：

- **Full Attention 层**：每步新生成的 K/V 都要追加保存，缓存随序列增长。
- **Linear Attention 层**：使用固定大小的两个状态——卷积滑动窗口（记录最近几帧的局部信息）和记忆矩阵（压缩所有历史 token 的全局信息）。

`__init__` 已完整提供，你只需实现两个方法：

- **缓存追加与读取**：首次写入时直接存入；此后每步将新内容沿序列方向追加到已有缓存末尾，并返回完整的历史序列供当前步使用。
- **查询已缓存长度**：返回当前缓存中已保存的序列长度，用于告知后续模块"历史到哪里了"。

---

### Task 2：Full Attention 层的 KV Cache (`tiny_inference/manual_attention.py`)

Full Attention 的 KV Cache 逻辑集中在一处：**在计算注意力之前，将本步产生的 K、V 追加到缓存，然后用完整的历史 K/V（而非仅本步的 K/V）参与后续计算。**

- Prefill 时，K/V 包含整个 prompt 的信息，是第一次写入缓存；
- Decode 每步，K/V 只有 1 个 token 的信息，追加后缓存变长；
- 不使用缓存时（no-cache 模式），直接使用当前步的 K/V 即可。

---

### Task 3：Linear Attention 层的缓存 (`tiny_inference/manual_linear.py`)

Linear Attention 的缓存涉及两个独立的实现点：

#### 3a：单步卷积更新

Decode 阶段每次只有 1 个新 token，不能对整段序列重新卷积。需要维护一个固定大小的滑动窗口：每步将新 token 滑入窗口右端，最旧的一帧从左端淘汰，然后基于这个窗口做卷积。函数签名和参数说明已在文件中提供。

#### 3b：在 `qwen3_5_linear_attn_forward` 中读写缓存状态

这是 Linear Attention 缓存的调度核心，需要实现四处位置：

1. **判断走哪条路径**：根据是否有缓存、序列长度、当前位置等条件，决定走"prefill 路径"还是"decode 路径"（即设置 `use_precomputed_states` 标志）。

2. **从缓存中读取本层状态**：在实际计算前，从缓存中取出该层上一步保存的滑动窗口和记忆矩阵，decode 路径会将其传给对应的计算函数。

3. **prefill 结束后保存卷积状态**：prefill 路径中，将当前序列末尾的若干帧保存为滑动窗口快照写入缓存，序列较短时需在左侧补零以对齐到 kernel_size。

4. **每步结束后保存递推状态**：无论 prefill 还是 decode，将更新后的记忆矩阵写回缓存，供下一步继续使用。

> 注：`torch_recurrent_gated_delta_rule`（单步递推）和 `torch_chunk_gated_delta_rule`（prefill 并行计算）均已完整提供，附有逐行注释，无需自行实现。

---

### Task 4：Forward 中创建并传递缓存 (`tiny_inference/manual_qwen3_5.py`)

本任务只需实现一处：**首次调用时创建缓存对象，并维护当前位置偏移**。

第一次 prefill 时，缓存对象还不存在，需要在这里创建。同时，从缓存中查询已有序列的长度（`past_seen_tokens`）——prefill 时为 0，decode 时为已生成的长度。后续代码会用这个偏移生成 `cache_position`（当前 batch 每个 token 在完整序列中的绝对位置），用于位置编码和注意力掩码的生成。

> 注：将缓存传入每一层和 forward 结束后返回缓存的代码已经写好——它们始终传递 `past_key_values` 这个变量。实现本 TODO 之前，该变量为 `None`，各层收到的是空缓存（等价于 no-cache）；实现后，变量指向真正的缓存对象，传递才真正生效。

---

### Task 5：Prefill + Decode 解码循环 (`tiny_inference/manual_decoding.py`)

这是最终将所有部分串联起来的地方。Phase 1 中每步都重复整段序列；Phase 2 中需要改为两阶段：

- **Prefill**：用完整 prompt 调用一次模型 forward（已提供），建立所有层的缓存，取最后一个位置的输出采样第一个 token。
- **Decode**：此后每步只将刚生成的 1 个 token 传入 forward，同时携带上一步返回的缓存；forward 内部更新缓存并返回新缓存，如此循环直到生成结束。

no-cache 路径（每步重新输入完整序列）已完整保留作为对照基线，仅需实现 `if use_cache:` 分支内的单步缓存路径。

---

## 如何定位需要填写的代码

在项目根目录执行：

```bash
grep -rn "TODO: KV Cache" tiny_inference/
```

这会列出所有需要实现的位置及其描述。

---

## 如何运行与验证

```bash
# 激活环境
uv sync && source .venv/bin/activate

# 运行完整测试（正确性 + 速度对比，默认行为）
python test_phase2.py

# 只跑正确性测试
python test_phase2.py --stage correctness

# 只跑速度对比
python test_phase2.py --stage speed
```


**验证要点**：

1. **模型能正常生成连贯文本**（测试脚本的正确性测试全部通过）。
2. **Decode 速度显著快于无缓存基线**：测试脚本会直接打印两种模式的对比，实现正确后 decode 速度应有明显提升。

---

## 提交要求

完成实现后，运行以下命令获取最终结果：

```bash
python test_phase2.py
```

正确性应全部 `[PASS]`；**Speed Comparison** 里 `with cache` 的 Decode 应明显快于 `no cache`。终端输出示例（数值因机器而异）：

```
============================================================
  Phase 2 KV Cache – Correctness Tests
============================================================
  [PASS] 1+1=2
  [PASS] Capital of France
  [PASS] Days in a week
  [PASS] Sky is blue
  [PASS] 3×4=12
  [PASS] Water is wet
------------------------------------------------------------
  Result: 6/6 passed
============================================================

============================================================
  Phase 2 KV Cache – Speed Comparison
============================================================
                                    with cache      no cache
  Prefill (tokens/s)                    xxx.xx        xxx.xx
  Decode  (tokens/s)                     xx.xx          x.xx
  Total elapsed (s)                       x.xxx        xx.xxx

  Decode speedup (cache / no-cache): x.xx×
  [OK] KV Cache decode is at least as fast as baseline.
============================================================
```

测试通过后：

1. **截图**：截屏保存 test_phase2.py 的运行结果（正确性测试通过且能看出 decode 较明显提速即可）。
2. **准备代码**：在项目根目录执行下述指令，会生成 **`<学号>_project_phase2/`**。
```
./submit.sh <学号>
```
3. **放入截图**：把截图放进同一文件夹。
4. **打包上传**：将该文件夹压缩为 **`<学号>_project_phase2.zip`**，按课程要求提交。

压缩包解压后应看到：5 个 `.py` + 截图，都在 `<学号>_project_phase2/` 根目录下。

---
