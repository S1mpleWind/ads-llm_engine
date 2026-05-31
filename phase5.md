# Phase 5：Client 端的 Context 管理 —— 多轮对话 Chatbox

> **前置要求**：建议先完成 [phase4.md](phase4.md)。Phase 5 将视角从「推理引擎内部」转向「调用方（client）」：用已有的 `TinyQwenEngine` 搭建一个多轮对话 chatbox，并在 client 端实现**上下文管理**。Phase 2~4 的引擎代码保持不变；Phase 5 仅新增 `tiny_inference/context.py`（你要实现的 TODO 都在这一个文件里，共 4 个标记块，对应 3 个任务）和一个零依赖的 Web UI `chat_app.py`（已实现）。

## 目录

- [Phase 5：Client 端的 Context 管理 —— 多轮对话 Chatbox](#phase-5client-端的-context-管理--多轮对话-chatbox)
  - [目录](#目录)
  - [如何把你的 Phase 4 实现合并到 Phase 5 分支](#如何把你的-phase-4-实现合并到-phase-5-分支)
  - [背景与动机](#背景与动机)
  - [架构总览](#架构总览)
  - [任务与实现要求](#任务与实现要求)
    - [Task 1：摘要压缩 `compress` (`tiny_inference/context.py`)](#task-1摘要压缩-compress-tiny_inferencecontextpy)
    - [Task 2：Prefix Cache Friendly 的 messages 拼装 `build_messages` (`tiny_inference/context.py`)](#task-2prefix-cache-friendly-的-messages-拼装-build_messages-tiny_inferencecontextpy)
    - [Task 3：会话持久化 `save` / `load` (`tiny_inference/context.py`)](#task-3会话持久化-save--load-tiny_inferencecontextpy)
  - [如何定位需要填写的代码](#如何定位需要填写的代码)
  - [如何运行与验证](#如何运行与验证)
    - [自动评测](#自动评测)
    - [跑起 Chatbox](#跑起-chatbox)
  - [提交要求](#提交要求)

---

## 如何把你的 Phase 4 实现合并到 Phase 5 分支

Phase 5 自身的实现任务（3 个 Task、4 个代码标记）**不依赖** Prefix Cache 是否实现——逻辑测试基于一个假 tokenizer，与真实模型及 Phase 3/4 代码完全解耦。但如果你想在 chatbox 里看到 **Prefix Cache 命中**的效果（`--prefix-cache`），就需要把 Phase 3/4 的实现合并进来：

```bash
git fetch origin
git checkout phase5            # 从 origin/phase5 创建本地分支
git merge phase4               # 合入你已完成的 Phase 3/4（如有 conflict 手动解决）
python test_phase4.py          # 确认引擎侧仍全部通过
```

---

## 背景与动机

Phase 2~4 优化的都是**引擎内部**；Phase 5 你换成**引擎的调用方**，用它搭一个多轮聊天的 chatbox。

核心问题是：**模型上下文窗口有限，而对话历史无限增长。** 每轮都要把「system prompt + 历史轮次 + 本轮输入」拼成 prompt，轮数越多 prompt 越长，迟早超出最大长度报错；即便没超，prefill 成本也随长度线性增长，对话轮数越多，速度越慢、成本越高。

所以交给引擎前，client 要先做上下文管理把 history 压缩。常见策略：

| 策略 | 思路 | 代价 |
|------|------|------|
| 截断 / 滑动窗口 | 直接丢最早的轮次 | 早期信息彻底丢失 |
| **摘要压缩（本 Phase 核心）** | 把最旧的若干轮**喂给模型自己**生成压缩摘要，用摘要替换原文 | 多一次摘要推理，但信息密度高 |
| 检索（RAG） | 历史进向量库，按需检索 | 超出本课范围 |

此外，client 的管理策略直接影响 server 端 Prefix Cache（Phase 3/4）的命中率：保持**稳定的 prompt prefix**，引擎只需 prefill 新增部分；若每轮都重写开头，prefix cache 就会失效，Phase 3/4 的优化随之白费。

---

## 架构总览

```
   浏览器 (chat_app.py 的 Web UI)
        │  POST /api/chat {session, message}
        ▼
   ┌──────────────────────── ContextManager（待实现）────────────────────────┐
   │  add_user(text)                                                            │
   │  maybe_compress(summarize_fn)   ── 超 budget？→ compress() 把最旧的轮折叠成摘要 │
   │  build_messages()               ── [system][summary][recent turns…]│
   │  add_assistant(reply) → save()  ── 落盘到 SessionStore（多会话、可持久化）   │
   └────────────────────────────────────────────────────────────────────────────┘
        │  messages（已压缩）
        ▼
   TinyQwenEngine.generate(messages)      ← Phase 2~4 的引擎，原样复用
        │
        ▼
   reply → 显示 + 更新右侧 Context 状态面板
```

`ContextManager` 维护单个会话的三件东西：固定的 `system_prompt`、把早期轮次压缩成的滚动 `summary`、以及尚未压缩的 `turns`。`SessionStore`（已实现）把多个会话以 JSON 落盘到一个目录，支持列出 / 切换 / 持久化。

---

## 任务与实现要求

所有需要你实现的位置都用以下标记包裹（共 **4 个标记块、对应 3 个 Task**，`save`/`load` 同属 Task 3）：

```python
# ===== TODO: Context - xxx (START) =====
...
# ===== TODO: Context - xxx (END) =====
```

文件里**已实现**的辅助方法（`count_tokens` / `token_budget` / `prompt_token_ids` / `maybe_compress` / `to_dict` / `_summary_message`）可以直接调用，请先读一遍它们的 docstring。

---

### Task 1：摘要压缩 `compress` (`tiny_inference/context.py`)

把除最近 `keep_recent_turns` 轮外、最旧的那批轮折叠进 `self.summary`，并从 `self.turns` 移除（没有可折叠的轮就直接返回）。摘要要**滚动累积**——把旧摘要连同被折叠轮的原文一起交给 `summarize_fn` 生成新摘要并覆盖 `self.summary`，这样更早的信息不会丢。最后维护压缩相关的统计字段。

> `summarize_fn`（在 `maybe_compress` 调用 `compress` 时传入）是注入进来的「摘要器」：chatbox 里接真实引擎（让模型自己写摘要），测试里是桩函数。这种依赖注入让 `compress` 既能跑真模型、又能被快速确定性地测试。

---

### Task 2：Prefix Cache Friendly 的 messages 拼装 `build_messages` (`tiny_inference/context.py`)

把当前会话状态拼成发给引擎的 `messages`，**顺序固定**：

```
[ system_prompt ]
[ summary（optional）]
[ turns 展开：user / assistant 交替 ]
```

返回 `list[dict]`，每项含 `role` 和 `content` 两个键。

---

### Task 3：会话持久化 `save` / `load` (`tiny_inference/context.py`)

会话历史是纯文本，用 **JSON** 落盘即可，可读、跨进程、重启不丢。

- **`save(path)`**：把 `to_dict()`（已实现）的结果写成 JSON。`path` 是文件路径，目录已由 `SessionStore` 创建，无需在此重复建目录。
- **`load(path, tokenizer)`**（`@classmethod`）：读回 JSON 重建 `ContextManager`，并还原 `summary` / `turns` / 统计字段；tokenizer 由调用方传入，不入盘。

多会话由 `SessionStore`（已实现）封装：不同 `session_id` 写到不同文件而已。

---

## 如何定位需要填写的代码

```bash
grep -rn "TODO: Context" tiny_inference/context.py
```

共 4 处标记块（`build_messages` 1 处、`compress` 1 处、`save` 1 处、`load` 1 处），对应上述 3 个 Task。

---

## 如何运行与验证

### 自动评测

```bash
uv sync && source .venv/bin/activate

# 默认：3 个不依赖模型的逻辑测试（快、确定性）——提交截图用这个
python test_phase5.py

# 单独某个
python test_phase5.py --stage compression
python test_phase5.py --stage prefix
python test_phase5.py --stage persistence

# 端到端：加载真实 Qwen3.5 引擎跑一段会被压缩的多轮对话（慢，需要权重）
python test_phase5.py --stage e2e

# 全部（含 e2e）
python test_phase5.py --stage all
```

**验证要点**：

1. **compression**：在小 budget 下连续对话 8 轮，应触发摘要压缩（`num_compressions ≥ 1`），最旧的轮被折叠进 `summary`、最近 `keep_recent_turns` 轮原文保留，最终回到 budget 内。
2. **prefix**：不触发压缩时，相邻两轮 prompt 的 shared prefix 应几乎等于上一轮全长（仅末尾的 generation 标记不同）——这正是 Prefix Cache 高命中的前提。
3. **persistence**：`save → load` 完整还原；`SessionStore` 能列出并读回多个会话。

终端输出示例：

```
============================================================
  Phase 5 – Context 压缩（summarization）
============================================================
  [PASS] 触发过摘要压缩 (num_compressions=3)
  [PASS] summary 已生成 (turns_folded=6)
  [PASS] 折叠+保留轮数守恒 (6+2=8)
  [PASS] 最近一轮原文被保留 (剩余 2 轮)
  [PASS] 最终在 budget 内 (tokens=38 <= budget=60)
  ...
  Overall: 3/3 stages passed
============================================================
```

### 跑起 Chatbox

```bash
python chat_app.py                       # 打开 http://127.0.0.1:8000
python chat_app.py --max-context 512 --reserve 128 --keep-recent 2   # 调小窗口，更容易看到压缩触发
python chat_app.py --prefix-cache        # 开 server 端 Prefix Cache（需已合并 Phase 3/4）
```

---

## 提交要求

完成实现后：

1. **截图**：保存 `python test_phase5.py`（默认 3 个逻辑测试全 PASS）的运行结果；如有条件，附一张 chatbox 触发压缩时的界面截图。
2. **准备代码**：在项目根目录执行：
   ```
   ./submit.sh <学号>
   ```
   将生成 `<学号>_project_phase5/` 目录，内含 `context.py` 与 `chat_app.py`。（若遇到权限错误，先执行 `chmod +x submit.sh`。）
3. **放入截图**：将截图放进同一目录。
4. **打包上传**：压缩为 `<学号>_project_phase5.zip`，按课程要求提交。

压缩包解压后应包含：`context.py` + `chat_app.py` + 截图，均位于 `<学号>_project_phase5/` 根目录下。

---
