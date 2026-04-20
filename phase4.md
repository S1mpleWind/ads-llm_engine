# Phase 4：Prefix Cache 向 SSD 的 Offloading

> **前置要求**：请先完成 [phase3.md](phase3.md)，并确保 `python test_phase3.py` 全部通过。Phase 4 在 Phase 3 的 `PrefixCache` 基础上扩展出一个**两级 Prefix Cache**：内存层（LRU）+ SSD 层（持久化）。Phase 3 的代码保留不动，Phase 4 新增的类与其接口兼容，可直接替换。

## 目录

- [Phase 4：Prefix Cache 向 SSD 的 Offloading](#phase-4prefix-cache-向-ssd-的-offloading)
  - [目录](#目录)
  - [如何将你的 Phase 3 实现合并到 Phase 4 分支](#如何将你的-phase-3-实现合并到-phase-4-分支)
  - [背景与动机](#背景与动机)
  - [架构总览](#架构总览)
  - [任务与实现要求](#任务与实现要求)
    - [Task 1：Cache 序列化 `to_cpu_state_dict` (`tiny_inference/cache.py`)](#task-1cache-序列化-to_cpu_state_dict-tiny_inferencecachepy)
    - [Task 2：Cache 反序列化 `from_cpu_state_dict` (`tiny_inference/cache.py`)](#task-2cache-反序列化-from_cpu_state_dict-tiny_inferencecachepy)
    - [Task 3：SSD 存储层 `DiskStore` (`tiny_inference/prefix_cache.py`)](#task-3ssd-存储层-diskstore-tiny_inferenceprefix_cachepy)
    - [Task 4：两级缓存查询 `TieredPrefixCache.lookup` (`tiny_inference/prefix_cache.py`)](#task-4两级缓存查询-tieredprefixcachelookup-tiny_inferenceprefix_cachepy)
    - [Task 5：两级缓存插入 `TieredPrefixCache.insert` (`tiny_inference/prefix_cache.py`)](#task-5两级缓存插入-tieredprefixcacheinsert-tiny_inferenceprefix_cachepy)
  - [如何定位需要填写的代码](#如何定位需要填写的代码)
  - [如何运行与验证](#如何运行与验证)
  - [提交要求](#提交要求)

---

## 如何将你的 Phase 3 实现合并到 Phase 4 分支

```bash
# 1. 拉取最新的 Phase 4 资料
git fetch origin
git checkout phase4            # 从 origin/phase4 创建本地分支

# 2. 合入你的 Phase 3 实现
git merge phase3               # 若出现 conflict，请手动解决

# 3. 验证 Phase 3 仍全部通过（Phase 4 的代码不应影响 Phase 3 行为）
python test_phase3.py
```

---

## 背景与动机

Phase 3 的 Prefix Cache 把所有快照都放在**内存**里。现实推理服务中，内存容量有限：

- Qwen3.5-0.8B 一条长 prompt 的缓存就已达数十 MB；
- 若服务每天接收数万个不同 system prompt，纯内存方案很快会爆。

**生产系统的通行做法是两级存储**（vLLM 的 CPU offload、SGLang 的 HiCache、DeepSeek 的 Context Caching 皆是此思路）：

- **Hot tier**：GPU/CPU 内存，容量小但访问快；
- **Cold tier**：SSD / 分布式存储，容量大但访问慢。

内存满时，把"最久未使用"的条目**下沉**到 SSD（而不是丢掉）；后续若再次命中，再从 SSD 读回内存。更进一步，SSD 上的快照**跨进程持久化**——重启引擎、切换 Python 解释器，之前见过的 prompt 仍然能命中。

Phase 4 的目标就是实现这样一个**两级 Prefix Cache**。

---

## 架构总览

```
   请求到来，lookup(token_ids)
             │
             ▼
   ┌───────────────────────┐      miss      ┌───────────────────────┐
   │   Memory (hot)        │ ─────────────► │   SSD (cold)          │
   │   OrderedDict (LRU)   │                │   DiskStore           │
   │   max_mem_entries=N   │ ◄───promote─── │   <root>/*.pt         │
   └───────────────────────┘                └───────────────────────┘
             │                                        ▲
             │ 满了按 LRU 驱逐 (offload)              │
             └────────────────────────────────────────┘

   进程启动时：DiskStore._rebuild_index() 扫 <root>/*.pt，恢复索引
   → 上个进程写过的快照，在本进程第一次 lookup 就能命中
```

关键设计：

1. **整 prompt 粒度**（与 Phase 3 一致）——每条记录的 key 是一段 prompt 的完整 token 序列。
2. **LRU 淘汰**——内存层溢出时 offload 队首（最久未使用）到 SSD；与 Phase 3 中 `PrefixCache` 的 LRU 策略一致。
3. **进程间持久化**——SSD 文件以 prompt tokens 的 SHA1 命名，启动时扫目录重建索引。
4. **跨 device 兼容**——落盘前 tensor 统一 `.cpu()`，加载时 `.to(device)`。

接口上，`TieredPrefixCache` 的 `lookup / insert` 与 Phase 3 的 `PrefixCache` **完全一致**，因此 `manual_decoding.py` 无需任何改动——engine 依据配置选择用哪个实现。

---

## 任务与实现要求

所有需要你实现的代码位置均使用以下标记包裹：

```python
# ===== TODO: SSD Offload - xxx (START) =====
...
# ===== TODO: SSD Offload - xxx (END) =====
```

---

### Task 1：Cache 序列化 `to_cpu_state_dict` (`tiny_inference/cache.py`)

把当前 `Qwen3_5DynamicCache` 的全部 tensor **拉到 CPU**，打包成一个普通 dict 供 `torch.save` 落盘。

1. 四个 tensor list（`key_cache` / `value_cache` / `conv_states` / `recurrent_states`）逐项处理：`None` 保持 `None`；Tensor 调 `.detach().cpu()`。
2. 三个元信息字段（`layer_types` / `transformer_layers` / `last_linear_layer`）直接复制。
3. **为什么必须 `.cpu()`？** GPU tensor 落盘会嵌入 device 信息，换机器或 GPU 不可用时加载会失败；且频繁往磁盘写显存没有意义——SSD 本身就是 CPU 侧设备。

---

### Task 2：Cache 反序列化 `from_cpu_state_dict` (`tiny_inference/cache.py`)

`@classmethod`，从 Task 1 产生的 dict 重建一个 `Qwen3_5DynamicCache`，并把所有 tensor 搬到目标 `device`。

1. 用 `object.__new__(cls)` 跳过 `__init__`（它需要 `config`，而落盘时并没有保存 config）——与 `clone()` 里的技巧相同。
2. 填回三个元信息 + 四个 tensor list；后者逐项 `None→None`、`Tensor→.to(device)`。

---

### Task 3：SSD 存储层 `DiskStore` (`tiny_inference/prefix_cache.py`)

共 3 个方法：

1. **`save(token_ids, cache)`**：以 `torch.save` 写出到 `<root>/<sha1>.pt`，文件内容 `{"tokens": [...], "state": cache.to_cpu_state_dict()}`；更新 `self._index[token_ids] = path` 与统计（`disk_writes`、`bytes_written`）。若同 key 文件已存在则跳过写入（内容相同）。
2. **`load(token_ids)`**：按索引找到文件路径 → `torch.load(..., map_location="cpu", weights_only=False)` → `Qwen3_5DynamicCache.from_cpu_state_dict(payload["state"], self.device)` → 更新 `disk_reads` → 返回。索引缺失或文件丢失返回 `None`。
3. **`_rebuild_index()`**（**进程间持久化的关键**）：遍历 `self.root_dir` 下所有 `.pt` 文件，读出 `tokens` 字段重建 `self._index`。**不要**把 `state` 也 load 进来——那会吃爆内存。坏文件要 try/except 住，不应让整个引擎起不来。

> 文件名为何用 SHA1？因为两个不同进程只要看到相同 `prompt_tokens`，就会算出相同文件名——这是"跨进程命中同一快照"的前提。

---

### Task 4：两级缓存查询 `TieredPrefixCache.lookup` (`tiny_inference/prefix_cache.py`)

语义与 Phase 3 `PrefixCache.lookup` 完全一致，只是查询范围扩展到两层：

1. 先扫内存 `self._mem.items()` 找最长前缀 `(best_mem_key, best_mem_len)`。
2. 再扫 SSD `self._disk.keys()` 找最长前缀 `(best_ssd_key, best_ssd_len)`。
3. 都没有 → `self.misses += 1; return (0, None)`。
4. 取更长者。若在内存：`move_to_end` 更新 LRU，`self.mem_hits += 1`。若在 SSD：调用已实现的 `self._promote_to_mem(key)`（它会先按需 LRU 驱逐内存队首到 SSD，再 load 回内存），`self.ssd_hits += 1`。
5. 完全匹配时 `matched_len` 截断为 `len - 1`（沿用 Phase 3 约束）。
6. `self.hits += 1; self.hit_tokens += matched_len`；返回 `(matched_len, cache.clone())`。

---

### Task 5：两级缓存插入 `TieredPrefixCache.insert` (`tiny_inference/prefix_cache.py`)

Prefill 结束后把 `(tokens, cache.clone())` 放到内存层，并维护 LRU：

1. key 已在内存 → `move_to_end` 后 return。
2. 内存满（`len >= max_mem_entries`）→ `popitem(last=False)` 取出队首，`self._disk.save(key, cache)` 下沉到 SSD，`self.offloads += 1`。
3. `self._mem[key] = cache.clone()`（新条目天然位于队尾）。

> 注意：`manual_decoding.py` 无需任何改动——它里面的两处 Phase 3 TODO 已经调用 `prefix_cache.lookup / insert`；两级缓存与纯内存缓存接口一致。

---

## 如何定位需要填写的代码

在项目根目录执行：

```bash
grep -rn "TODO: SSD Offload" tiny_inference/
```

共 5 处 TODO（cache.py 2 处 + prefix_cache.py 3 处），分别对应上述 5 个 Task。

---

## 如何运行与验证

```bash
# 激活环境（若尚未安装依赖）
uv sync && source .venv/bin/activate

# 完整测试（正确性 + 持久化 + 速度对比）
python test_phase4.py

# 仅某一阶段
python test_phase4.py --stage correctness
python test_phase4.py --stage persistence
python test_phase4.py --stage speed
```

**验证要点**：

1. **正确性**：内存层容量设为 1，连续请求 3 个不同 prompt，验证第 1 个被挤到 SSD 后再次请求仍能命中（`ssd_hits > 0`），输出关键词仍然正确。
2. **持久化**：脚本会跑完一轮 → 销毁 engine → 在同一目录下**新建**另一个 engine（模拟进程重启）→ 请求同一个 prompt，应直接命中 SSD（`ssd_hits > 0`）。
3. **速度**：两级缓存二次请求的 prefill 耗时应显著低于无缓存 baseline；SSD 命中因多了一次 `torch.load` + `.to(device)`，会比内存命中稍慢，但仍远快于从头 prefill。

> **关于数值一致性**：与 Phase 3 同样存在 $10^{-5}$ 级数值差异（linear attention 的 chunk-prefill 与 recurrent-decode 路径在有限精度下不完全等价），贪心解码偶尔会因此翻动首 token。这是混合架构下 partial prefill 的**固有特性**。

终端输出示例（数值因机器而异）：

```
============================================================
  Phase 4 SSD Offload – Correctness + Eviction
============================================================
  [PASS] Capital of France (mem=1, ssd=0 after turn 1)
  [PASS] Days in a week    (mem=1, ssd=1 after turn 2, evicted 1)
  [PASS] Sky is blue       (mem=1, ssd=2 after turn 3, evicted 2)
  [PASS] Re-query evicted  (ssd_hits=1)
  Result: 4/4 passed
============================================================
  Phase 4 SSD Offload – Cross-Process Persistence
============================================================
  Writer engine finished, entries on disk: 2
  Reader engine (fresh instance) lookup → ssd_hits=1  [OK]
============================================================
  Phase 4 SSD Offload – Speed Comparison
============================================================
                               mem hit   ssd hit   no cache
  Prefill elapsed (s)          0.0217    0.0612    1.8812
  Speedup vs no-cache          86.7×     30.7×
============================================================
```

---

## 提交要求

完成实现后：

1. **截图**：保存 `python test_phase4.py` 的运行结果。
2. **准备代码**：在项目根目录执行：
   ```
   ./submit.sh <学号>
   ```
   将生成 `<学号>_project_phase4/` 目录，内含 3 个 `.py` 文件（`cache.py`、`prefix_cache.py`、`engine.py`）。
3. **放入截图**：将截图放置于同一目录内。
4. **打包上传**：将该目录压缩为 `<学号>_project_phase4.zip`，按课程要求提交。

压缩包解压后应包含：3 个 `.py` 文件 + 截图，均位于 `<学号>_project_phase4/` 根目录下。

---
