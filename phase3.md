# Phase 3：用 Prefix Cache 实现跨请求复用

> **前置要求**：请先完成 [phase2.md](phase2.md)，并确保 `python test_phase2.py` 全部通过，且 decode 速度相较 no-cache 有明显提升。Phase 3 将在 Phase 2 已实现的 KV Cache 基础上继续扩展，**不会重复实现 Phase 2 的 TODO**——请先将你的 Phase 2 实现合并到本分支后再开始。

## 目录

- [Phase 3：用 Prefix Cache 实现跨请求复用](#phase-3用-prefix-cache-实现跨请求复用)
  - [目录](#目录)
  - [如何将你的 Phase 2 实现合并到 Phase 3 分支](#如何将你的-phase-2-实现合并到-phase-3-分支)
  - [任务总览](#任务总览)
  - [任务与实现要求](#任务与实现要求)
    - [Task 1：Cache 深拷贝 `clone()` (`tiny_inference/cache.py`)](#task-1cache-深拷贝-clone-tiny_inferencecachepy)
    - [Task 2：Prefix Cache查询 `PrefixCache.lookup` (`tiny_inference/prefix_cache.py`)](#task-2prefix-cache查询-prefixcachelookup-tiny_inferenceprefix_cachepy)
    - [Task 3：Prefix Cache插入 `PrefixCache.insert` (`tiny_inference/prefix_cache.py`)](#task-3prefix-cache插入-prefixcacheinsert-tiny_inferenceprefix_cachepy)
    - [Task 4：Prefill 前查询Prefix Cache (`tiny_inference/manual_decoding.py`)](#task-4prefill-前查询prefix-cache-tiny_inferencemanual_decodingpy)
    - [Task 5：Prefill 后写入Prefix Cache (`tiny_inference/manual_decoding.py`)](#task-5prefill-后写入prefix-cache-tiny_inferencemanual_decodingpy)
  - [如何定位需要填写的代码](#如何定位需要填写的代码)
  - [如何运行与验证](#如何运行与验证)
  - [提交要求](#提交要求)

---

## 如何将你的 Phase 2 实现合并到 Phase 3 分支

```bash
# 1. 拉取最新的 Phase 3 资料
git fetch origin
git checkout phase3            # 从 origin/phase3 创建本地分支

# 2. 合入你的 Phase 2 实现
git merge phase2               # 若出现 conflict，请手动解决，确保你在 Phase 2 中的实现正确合并到 Phase 3 分支

# 3. 验证 Phase 2 仍全部通过
python test_phase2.py
```

## 任务总览

Phase 2 的 KV Cache 解决了**单次请求内部**「每生成一个 token 就要对整段序列重新计算」的问题。但在实际推理服务中，**跨请求**之间仍存在大量重复工作：

- 大量请求共享同一段 **system prompt**（例如「你是一个简洁的助手，只回答问题……」）；
- 多轮对话中，第 $k$ 轮的输入实际上是「前 $k-1$ 轮的全部内容 + 一条新的用户消息」，而前 $k-1$ 轮的内容此前已完整 prefill 过；
- 在 RAG / Agent 场景下，一条较长的「背景资料 + 工具描述」会被频繁复用。

Phase 2 的缓存在每次请求结束后即被丢弃。**Prefix Cache**（Prefix Cache，又称 Prompt Cache 或 KV Cache Reuse）的目标是将「prefill 结束时的缓存」跨请求保存下来，使新请求**只需 prefill「自身多出的 Suffix 部分」**，Prefix 部分无需重复计算。

Phase 3 将实现**整 prompt 粒度的Prefix Cache**——这是最简单也最常见的形式：

| 操作                  | 描述                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| **Lookup**            | 新 prompt 到来时，在已保存的快照中查找「最长的、能作为新 prompt Prefix」的记录 |
| **Partial Prefill**   | 命中时，加载该快照作为 `past_key_values`，仅将新 prompt 去掉 Prefix 后的 Suffix tokens 送入 forward |
| **Insert**            | prefill 结束后，将当前 Cache 的深拷贝与对应的 prompt token 序列一并存入Prefix Cache，供后续请求复用 |

这意味着，对于重复的 prompt，**第二次请求的 prefill 阶段几乎可以瞬时完成**（仅需处理最后一个 token 以生成首个输出 logits）。

> 为什么Prefix Cache必须使用**深拷贝**？
> 快照被保存后，会被**后续请求**继续追加 K/V、原地更新 `conv_state` / `recurrent_state`。  
> 若存储的是原对象的引用，第二次复用时获取的将不再是「prefill 刚结束时的状态」，  
> 而是混杂了上一次请求若干步 decode 的脏数据，这会导致结果错误。

---

## 任务与实现要求

所有需要你实现的代码位置均使用以下标记包裹：

```python
# ===== TODO: Prefix Cache - (START) =====
...
# ===== TODO: Prefix Cache - (END) =====
```

---

### Task 1：Cache 深拷贝 `clone()` (`tiny_inference/cache.py`)

`Qwen3_5DynamicCache.clone()` 应返回一个与当前 Cache 结构完全相同、但所有 tensor 均已独立拷贝的新缓存对象。Prefix Cache的**插入**与**查询**操作均会调用此方法。

- 新实例可跳过 `__init__`（该方法需要 `config` 参数），使用 `object.__new__(Qwen3_5DynamicCache)` 构造一个"空壳"，然后将 `layer_types` / `transformer_layers` / `last_linear_layer` 直接复制过去。
- 四个列表 `key_cache` / `value_cache` / `conv_states` / `recurrent_states` 需逐项处理：`None` 保持为 `None`，Tensor 则调用 `.clone()`。

---

### Task 2：Prefix Cache查询 `PrefixCache.lookup` (`tiny_inference/prefix_cache.py`)

在已保存的 `(prompt_tokens, cache_snapshot)` 记录中，查找「最长的、能作为当前 `token_ids` Prefix」的记录，返回 `(matched_len, cloned_cache)`。

1. **Prefix**指 `cached_tokens == token_ids[:len(cached_tokens)]`——请注意方向不要写反。
2. 当完全匹配（`matched_len == len(token_ids)`）时，**需主动将 matched_len 截断为 len - 1**。否则 forward 将收到长度为 0 的 `input_ids`，无法生成首个输出 logits。
3. 命中时返回的 cache 必须是通过 `cached_cache.clone()` 获得的副本，不可直接返回原对象——同一条快照可能被多次复用，每次复用都会对其进行 K/V 追加。
4. 查询结束后需正确更新 `self.hits` / `self.misses` / `self.hit_tokens`（测试脚本会读取这些统计信息）。
5. **LRU 维护**：命中后需将该条记录通过 `OrderedDict.move_to_end` 移到队尾，标记为「最近使用」。否则 insert 触发淘汰时，可能把仍在被频繁命中的热点前缀丢掉。

---

### Task 3：Prefix Cache插入 `PrefixCache.insert` (`tiny_inference/prefix_cache.py`)

在一次 prefill 结束后，将 `(prompt_token_ids, current_cache_snapshot)` 追加到记录中。

1. 存入的必须是 `cache.clone()` 的副本——理由与 lookup 一致：后续 decode 循环会持续向原 cache 写入 token，原对象将立即被"污染"。
2. 若已存在相同 key 的记录（即完全相同的 prompt 已存储），则**不重新 clone**，但需 `move_to_end` 更新 LRU 顺序（这条前缀刚刚又被写了一次，应视为最近使用）。
3. 采用 **LRU 淘汰策略**：超出 `max_entries` 时，用 `self._entries.popitem(last=False)` 丢弃「最久未使用」的一条记录，并把 `self.evictions` 加 1。

---

### Task 4：Prefill 前查询Prefix Cache (`tiny_inference/manual_decoding.py`)

位置在 `decode_tokens_manual` 函数中、首次调用 `qwen3_5_text_forward` 之前。

1. 检查是否需要启用Prefix Cache路径：`use_cache and prefix_cache is not None`。
2. 调用 `prefix_cache.lookup(input_ids[0].tolist())`，获取 `(matched_len, loaded_cache)`。
3. 命中时，将即将送入 forward 的 `prefill_input_ids` 截断为 `input_ids[:, matched_len:]`，并将 `prefill_past_kv` 设置为 `loaded_cache`。`attention_mask` 无需截断——它表示「完整序列」的长度，forward 内部会根据缓存长度正确生成 `cache_position`（这正是 Phase 2 中你已实现的 `past_seen_tokens` 分支逻辑）。
4. 将命中长度记录到 `prefix_hit_tokens`，后续用于 `timing` 展示。

未命中或未启用Prefix Cache时，保持原有行为：整段 prompt 送入 forward，`past_key_values=None`。

> 为何无需修改 `attention_mask`？Phase 2 已实现 forward 根据 `past_seen_tokens` 偏移生成 `cache_position`；同时 `create_causal_mask` 接收完整 `attention_mask` + 缓存对象，会自动为「历史 + 当前」构造正确的掩码。Phase 3 仅将 `past_seen_tokens` 从 0 变为 `matched_len`，其余逻辑均已就绪。

---

### Task 5：Prefill 后写入Prefix Cache (`tiny_inference/manual_decoding.py`)

位置在 prefill 结束后、首个 token sampled 之前（或之后，只要尚未进入 decode 循环均可）。

Prefill 刚结束时，`past_key_values` 恰好是「整段 prompt 处理完毕后的 Cache 状态」，这正是下一次相同 prompt 希望复用的状态。一旦进入 decode 循环，该状态即被修改。因此，应在此刻调用 `prefix_cache.insert(input_ids[0].tolist(), past_key_values)`（`insert` 内部会负责深拷贝）。

> 注：`decode_stream_manual` 中同样包含这两处逻辑，但其实现已预先写好（与 `decode_tokens_manual` 完全同构）。你只需将 `decode_tokens_manual` 中的两处 TODO 正确实现即可。

---

## 如何定位需要填写的代码

在项目根目录执行：

```bash
grep -rn "TODO: Prefix Cache" tiny_inference/
```

共 5 处 TODO，分别对应上述 5 个 Task。

---

## 如何运行与验证

```bash
# 激活环境（若尚未安装依赖）
uv sync && source .venv/bin/activate

# 完整测试（正确性 + 速度对比，默认行为）
python test_phase3.py

# 仅运行正确性测试
python test_phase3.py --stage correctness

# 仅运行速度对比
python test_phase3.py --stage speed
```

**验证要点**：

1. **正确性测试全部通过**：同一 prompt 连续请求两次，第二次必须 (a) 命中Prefix Cache（`prefix_hit_tokens > 0`），且 (b) 输出包含预期关键词。
2. **速度对比**：对于同一长 prompt，**第二次**请求在开启Prefix Cache时的 prefill 耗时，应显著低于关闭Prefix Cache时；测试脚本会打印加速比并给出 `[OK]` / `[WARN]` 判定。

> **关于数值一致性**：命中Prefix Cache后的输出**不一定**与「不使用Prefix Cache、从头 prefill」的结果逐字节相等。Qwen3.5 的 linear attention 在 prefill 时采用 `torch_chunk_gated_delta_rule`（分块并行），而 decode 单步则采用 `torch_recurrent_gated_delta_rule`（逐步递推）。两条路径在数学上等价，但在有限精度下会产生 $10^{-5}$ 量级的数值差异；贪心解码对 logits 顶端极为敏感，这种微小差异有时会导致首个 token 发生变化（例如 `Blue` ↔ `**Blue**`）。这是混合架构下 partial prefill 的**固有特性**，并非你的实现存在 bug——业界方案（如 vLLM）同样会遇到此问题，且通常被视为可接受的代价。

终端输出示例（数值因机器而异）：

```
============================================================
  Phase 3 Prefix Cache – Correctness Tests
============================================================
  [PASS] Capital of France  (prefix_hit_tokens=23)
  [PASS] Days in a week     (prefix_hit_tokens=24)
  [PASS] Sky is blue        (prefix_hit_tokens=24)
------------------------------------------------------------
  Result: 3/3 passed
============================================================

============================================================
  Phase 3 Prefix Cache – Speed Comparison
============================================================
  Prompt length (tokens): 97
  Prefix hit tokens (second call, Prefix Cache): 96

                                   w/ Prefix Cache   no Prefix Cache
  Prefill elapsed (s)                      0.0214            1.8732
  Total elapsed (s)                        0.7103            2.5428

  Prefill speedup (second call, cache / no-cache): 87.53×
  [OK] Prefix Cache 显著加速了第二次相同 prompt 的 prefill。
============================================================
```

---

## 提交要求

完成实现后：

1. **截图**：保存 `python test_phase3.py` 的运行结果（正确性全部通过，且能看到速度对比表格）。
2. **准备代码**：在项目根目录执行：
   ```
   ./submit.sh <学号>
   ```
   将生成 `<学号>_project_phase3/` 目录，内含 4 个你需要修改或新建的 `.py` 文件（`cache.py`、`prefix_cache.py`、`manual_decoding.py`、以及 `engine.py`）。
3. **放入截图**：将截图放置于同一目录内。
4. **打包上传**：将该目录压缩为 `<学号>_project_phase3.zip`，按课程要求提交。

压缩包解压后应包含：4 个 `.py` 文件 + 截图，均位于 `<学号>_project_phase3/` 根目录下。

---