# Phase 5：Client 端的 Context 管理（多轮对话 Chatbox）

> **前置要求**：建议先完成 [phase4.md](phase4.md)（或至少 [phase3.md](phase3.md)）。Phase 5 把视角从「推理引擎内部」切换到「调用方（client）」：用现成的 `TinyQwenEngine` 搭一个多轮对话 chatbox，并在 client 端实现**上下文管理**。Phase 2~4 的引擎代码保持不动；Phase 5 只新增一个 `tiny_inference/context.py`（你要填的 3 处 TODO 都在这里）和一个零依赖的 Web UI `chat_app.py`（已实现）。

## 目录

- [Phase 5：Client 端的 Context 管理（多轮对话 Chatbox）](#phase-5client-端的-context-管理多轮对话-chatbox)
  - [目录](#目录)
  - [如何把你的 Phase 4 实现合并到 Phase 5 分支](#如何把你的-phase-4-实现合并到-phase-5-分支)
  - [背景与动机](#背景与动机)
  - [架构总览](#架构总览)
  - [任务与实现要求](#任务与实现要求)
    - [Task 1：摘要压缩 `compress` (`tiny_inference/context.py`)](#task-1摘要压缩-compress-tiny_inferencecontextpy)
    - [Task 2：前缀友好的 messages 拼装 `build_messages` (`tiny_inference/context.py`)](#task-2前缀友好的-messages-拼装-build_messages-tiny_inferencecontextpy)
    - [Task 3：会话持久化 `save` / `load` (`tiny_inference/context.py`)](#task-3会话持久化-save--load-tiny_inferencecontextpy)
  - [如何定位需要填写的代码](#如何定位需要填写的代码)
  - [如何运行与验证](#如何运行与验证)
    - [自动评测](#自动评测)
    - [跑起 Chatbox（可选，但强烈建议）](#跑起-chatbox可选但强烈建议)
  - [提交要求](#提交要求)

---

## 如何把你的 Phase 4 实现合并到 Phase 5 分支

Phase 5 自身的 3 个 TODO **不依赖** Prefix Cache 是否实现——逻辑测试用一个假 tokenizer 跑，与模型、与 Phase 3/4 完全解耦。但如果你想在 chatbox 里看到 **Prefix Cache 命中**的效果（`--prefix-cache`），就需要把 Phase 3/4 的实现合并进来：

```bash
git fetch origin
git checkout phase5            # 从 origin/phase5 创建本地分支
git merge phase4               # 合入你已完成的 Phase 3/4（如有 conflict 手动解决）
python test_phase4.py          # 确认引擎侧仍全部通过
```

---

## 背景与动机

Phase 2~4 优化的都是**引擎内部**：KV Cache（单请求内复用）、Prefix Cache（跨请求复用）、SSD offload（冷缓存下沉）。Phase 5 换一个身份——**你是引擎的调用方**，要用它搭一个能多轮聊天的 chatbox。

刚一上手就会撞到一个本质矛盾：

> **模型的上下文窗口是有限的，而对话历史是无限增长的。**

每轮对话，client 都得把「system 提示 + 之前所有轮次 + 本轮输入」拼成一个 prompt 发给引擎。轮数越多，prompt 越长：

- 迟早超过模型最大上下文长度，请求直接报错；
- 即便没超，prefill 成本随 prompt 长度线性增长，越聊越慢、越聊越贵。

所以**在把 prompt 交给引擎之前**，client 必须先做一轮上下文管理，把历史压进预算。常见策略：

| 策略 | 思路 | 代价 |
|------|------|------|
| 截断 / 滑动窗口 | 直接丢最早的轮次 | 早期信息彻底丢失 |
| **摘要压缩（本 Phase 核心）** | 把最旧的若干轮**喂给模型自己**生成压缩摘要，用摘要替换原文 | 多一次摘要推理，但信息密度高 |
| 检索（RAG） | 历史进向量库，按需检索 | 超出本课范围 |

**还有一个常被忽视、但本 Phase 特别强调的点**：client 端的管理策略会直接影响 server 端 Prefix Cache（Phase 3/4）的命中率。写得好的管理器让连续两轮请求共享一段**稳定的 prompt 前缀**，引擎只需 prefill 新增的一小段；写得差的管理器（比如每轮都重写开头）会让前缀缓存每轮失效——client 的一个设计失误，能把你 Phase 3/4 辛苦做的引擎优化全部抵消。

---

## 架构总览

```
   浏览器 (chat_app.py 的 Web UI)
        │  POST /api/chat {session, message}
        ▼
   ┌──────────────────────── ContextManager（你要实现）────────────────────────┐
   │  add_user(text)                                                            │
   │  maybe_compress(summarize_fn)   ── 超预算？→ compress() 把最旧的轮折叠成摘要 │
   │  build_messages()               ── [system][summary][recent turns…] 稳定前缀│
   │  add_assistant(reply) → save()  ── 落盘到 SessionStore（多会话、可持久化）   │
   └────────────────────────────────────────────────────────────────────────────┘
        │  messages（已压进预算 + 前缀稳定）
        ▼
   TinyQwenEngine.generate(messages)      ← Phase 2~4 的引擎，原样复用
        │  （若开了 --prefix-cache，则共享前缀直接命中，跳过重复 prefill）
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

把「最旧的若干轮」折叠进 `self.summary`，并从 `self.turns` 移除它们。

1. **选哪些轮折叠**：除最近 `keep_recent_turns` 轮外、最旧的那一批。
   `n_fold = len(self.turns) - self.keep_recent_turns`；若 `n_fold <= 0` 直接 `return`。
2. `fold, self.turns = self.turns[:n_fold], self.turns[n_fold:]`。
3. **拼摘要素材** `material`：若已有旧摘要先写进去（这样摘要是**滚动累积**的，不丢更早信息），再把 `fold` 里每轮的 `用户：.../助手：...` 追加上。
4. `prompt = _SUMMARY_INSTRUCTION + material`，调 `summarize_fn(prompt).strip()` 得到新摘要，覆盖 `self.summary`。
5. 维护统计：`num_compressions += 1`、`turns_folded += n_fold`。

> `summarize_fn` 是注入进来的「摘要器」：chatbox 里它被接到真实引擎（让模型自己写摘要），测试里是一个桩函数。这种依赖注入让 `compress` 既能跑真模型、又能被快速确定性地测试。

---

### Task 2：前缀友好的 messages 拼装 `build_messages` (`tiny_inference/context.py`)

把当前会话状态拼成发给引擎的 `messages`，**顺序固定**：

```
[ system_prompt ]            ← 永远最前，固定不变
[ summary（若有）]            ← 紧随其后；只有 compress() 真正发生时才会变
[ turns 展开：user/assistant ] ← 最新对话，逐轮追加在末尾
```

实现本身只有几行（system → `_summary_message()` → 逐个 `turn.to_messages()`），但**为什么是这个顺序**才是本 Task 的考点：

- Prefix Cache 按「token 前缀是否完全相同」复用 prefill。把最稳定的内容（system、已冻结的 summary）放最前、新增内容追加末尾，连续两轮就能共享一段尽可能长的前缀。
- 反例：每轮重新生成摘要、或把易变内容放开头 → 前缀每轮变 → Prefix Cache 每轮失效。这也是为什么 `maybe_compress`（已实现）**只在超预算时才压**，而不是每轮都压——让 summary 尽量稳定。

`prefix` 测试会度量相邻两轮 prompt 的「共享前缀长度」，验证你的拼装在不触发压缩时几乎完整复用了上一轮的 prompt。

---

### Task 3：会话持久化 `save` / `load` (`tiny_inference/context.py`)

会话历史是纯文本（不像 KV Cache 是 tensor），用 **JSON** 落盘即可，可读、跨进程、重启不丢。

- **`save(path)`**：建父目录 → `json.dump(self.to_dict(), ...)`（`ensure_ascii=False, indent=2`）。`to_dict()` 已实现。
- **`load(path, tokenizer)`**（`@classmethod`）：读 JSON → 用其中的配置字段构造一个 `cls(...)`（tokenizer 由调用方传入，不入盘）→ 填回 `summary` / `turns` / 统计字段 → 返回。

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

1. **compression**：小预算下连聊 8 轮，应触发摘要压缩（`num_compressions ≥ 1`），最旧的轮被折叠进 `summary`、最近 `keep_recent_turns` 轮原文保留，最终回到预算内。
2. **prefix**：不触发压缩时，相邻两轮 prompt 的共享前缀应几乎等于上一轮全长（仅末尾的 generation 标记不同）——这正是 Prefix Cache 高命中的前提。
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
  [PASS] 最终在预算内 (tokens=38 <= budget=60)
  ...
  Overall: 3/3 stages passed
============================================================
```

### 跑起 Chatbox（可选，但强烈建议）

亲手聊几句，比看测试输出更能体会 context 管理在干什么：

```bash
python chat_app.py                       # 打开 http://127.0.0.1:8000
python chat_app.py --max-context 512 --reserve 128 --keep-recent 2   # 调小窗口，更容易看到压缩触发
python chat_app.py --prefix-cache        # 开 server 端 Prefix Cache（需已合并 Phase 3/4）
```

右侧面板会实时显示当前 prompt 的 token 数 / 预算、压缩次数、滚动摘要正文，开了 `--prefix-cache` 还能看到「本轮命中省下的 token 数」。试着多聊几轮把窗口顶满，观察摘要是如何把老对话压下去、而最近几轮原文被保留的。

---

## 提交要求

完成实现后：

1. **截图**：保存 `python test_phase5.py`（默认 3 个逻辑测试全 PASS）的运行结果；如有条件，附一张 chatbox 触发压缩时的界面截图。
2. **准备代码**：在项目根目录执行：
   ```
   ./submit.sh <学号>
   ```
   将生成 `<学号>_project_phase5/` 目录，内含 `context.py` 与 `chat_app.py`。
3. **放入截图**：将截图放进同一目录。
4. **打包上传**：压缩为 `<学号>_project_phase5.zip`，按课程要求提交。

压缩包解压后应包含：`context.py` + `chat_app.py` + 截图，均位于 `<学号>_project_phase5/` 根目录下。

---
