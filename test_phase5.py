"""
Phase 5 自动评测：Client 端 Context 管理

用法
----
    python test_phase5.py                 # 默认：3 个不依赖模型的逻辑测试（快、确定性）
    python test_phase5.py --stage compression
    python test_phase5.py --stage prefix
    python test_phase5.py --stage persistence
    python test_phase5.py --stage e2e      # 端到端：加载真实 Qwen3.5 引擎跑多轮对话（慢）
    python test_phase5.py --stage all       # 上面全部（含 e2e）

三个逻辑测试用一个**假 tokenizer**（按词切分）驱动，既快又确定，专门检验你在
`tiny_inference/context.py` 里实现的三处 TODO：
    compression  → Task 1  ContextManager.compress
    prefix       → Task 2  ContextManager.build_messages（前缀缓存友好）
    persistence  → Task 3  ContextManager.save / load（+ SessionStore）

e2e 用真实引擎跑一段会被压缩的多轮对话，验证整套 client 管理能驱动真模型。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from tiny_inference.context import ContextManager, SessionStore, Turn

PASS = "[PASS]"
FAIL = "[FAIL]"


# ---------------------------------------------------------------------------
# 假 tokenizer：按空格切词，token id 由一个稳定 vocab 决定。
# 相同文本 → 相同 id 序列，因此「共享前缀长度」可以反映文本级别的前缀复用。
# ---------------------------------------------------------------------------
class FakeTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab) + 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        toks: list[str] = []
        for m in messages:
            toks.append(f"<{m['role']}>")
            toks.extend(str(m["content"]).split())
        if add_generation_prompt:
            toks.append("<gen>")
        ids = [self._id(t) for t in toks]
        return ids if tokenize else " ".join(toks)


def _shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _header(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ===========================================================================
# Stage 1: compression（Task 1）
# ===========================================================================
def stage_compression() -> bool:
    _header("Phase 5 – Context 压缩（summarization）")
    tok = FakeTokenizer()
    cm = ContextManager(
        tok,
        system_prompt="你是助手",
        max_context_tokens=70,
        reserve_for_reply=10,
        keep_recent_turns=2,
    )

    calls = {"n": 0}

    def stub_summarize(prompt: str) -> str:
        calls["n"] += 1
        # 摘要必须比原文短，否则压不下去。这里返回一小段固定摘要。
        return f"早期要点摘要v{calls['n']}：用户与助手讨论过若干话题。"

    n_turns = 8
    for i in range(n_turns):
        cm.add_user(f"这是用户的第 {i} 条比较长的提问 包含 一些 额外 的 词 来 占 token")
        cm.maybe_compress(stub_summarize)
        cm.add_assistant(f"这是助手对第 {i} 条的简短回复")

    ok = True

    cond = cm.num_compressions >= 1
    ok &= cond
    print(f"  {PASS if cond else FAIL} 触发过摘要压缩 (num_compressions={cm.num_compressions})")

    cond = cm.summary is not None and len(cm.summary) > 0
    ok &= cond
    print(f"  {PASS if cond else FAIL} summary 已生成 (turns_folded={cm.turns_folded})")

    # 折叠 + 保留 的轮数应等于总轮数
    cond = cm.turns_folded + len(cm.turns) == n_turns
    ok &= cond
    print(f"  {PASS if cond else FAIL} 折叠+保留轮数守恒 ({cm.turns_folded}+{len(cm.turns)}={n_turns})")

    # 最近的若干轮原文必须保留（没有被摘要掉）
    cond = any(f"第 {n_turns - 1} 条" in t.user for t in cm.turns)
    ok &= cond
    print(f"  {PASS if cond else FAIL} 最近一轮原文被保留 (剩余 {len(cm.turns)} 轮)")

    # 压缩后应回到预算之内（或已压到只剩 keep_recent_turns 轮的下限）
    final = cm.count_tokens(cm.build_messages())
    cond = final <= cm.token_budget() or len(cm.turns) <= cm.keep_recent_turns
    ok &= cond
    print(f"  {PASS if cond else FAIL} 最终在预算内 (tokens={final} <= budget={cm.token_budget()})")

    print(f"  Result: {'all passed' if ok else 'FAILED'}")
    return ok


# ===========================================================================
# Stage 2: prefix（Task 2）—— 前缀缓存友好
# ===========================================================================
def stage_prefix() -> bool:
    _header("Phase 5 – 前缀稳定性（与 Prefix Cache 协同）")
    tok = FakeTokenizer()
    # 预算给足，前几轮不触发压缩 —— 这样能纯粹检验 build_messages 的拼装顺序。
    cm = ContextManager(
        tok,
        system_prompt="你是一个有帮助的助手",
        max_context_tokens=100000,
        reserve_for_reply=256,
        keep_recent_turns=99,
    )

    ok = True
    prev_prompt: list[int] | None = None
    for i in range(4):
        cm.add_user(f"用户第 {i} 轮的问题 内容 若干 词")
        prompt = cm.prompt_token_ids(cm.build_messages())  # 本轮真正发给引擎的输入
        if prev_prompt is not None:
            shared = _shared_prefix_len(prev_prompt, prompt)
            # 没有压缩发生时，新一轮 prompt 应几乎完整复用上一轮 prompt（仅末尾 gen 标记不同）。
            cond = shared >= len(prev_prompt) - 3
            ok &= cond
            print(
                f"  {PASS if cond else FAIL} 轮 {i-1}→{i} 共享前缀 {shared}/{len(prev_prompt)} "
                f"(越接近越能命中 Prefix Cache)"
            )
        cm.add_assistant(f"助手第 {i} 轮的回复")
        prev_prompt = cm.prompt_token_ids(cm.build_messages())

    # 顺序检查：system 必须在最前，summary（若有）紧随其后
    msgs = cm.build_messages()
    cond = msgs[0]["role"] == "system" and "助手" in msgs[0]["content"]
    ok &= cond
    print(f"  {PASS if cond else FAIL} system 提示位于 prompt 最前端")

    print(f"  Result: {'all passed' if ok else 'FAILED'}")
    return ok


# ===========================================================================
# Stage 3: persistence（Task 3）
# ===========================================================================
def stage_persistence() -> bool:
    _header("Phase 5 – 会话持久化（save / load / SessionStore）")
    tok = FakeTokenizer()
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        cm = ContextManager(
            tok,
            system_prompt="持久化测试系统提示",
            max_context_tokens=512,
            reserve_for_reply=64,
            keep_recent_turns=2,
        )
        cm.summary = "这是一段已有摘要"
        cm.turns = [Turn("问题A", "回答A"), Turn("问题B", "回答B")]
        cm.num_compressions = 3
        cm.turns_folded = 5

        # 直接 save/load 往返
        path = os.path.join(tmp, "direct.json")
        cm.save(path)
        cond = os.path.exists(path)
        ok &= cond
        print(f"  {PASS if cond else FAIL} save 写出了 JSON 文件")

        loaded = ContextManager.load(path, tok)
        cond = (
            loaded.system_prompt == cm.system_prompt
            and loaded.summary == cm.summary
            and loaded.max_context_tokens == cm.max_context_tokens
            and loaded.reserve_for_reply == cm.reserve_for_reply
            and loaded.keep_recent_turns == cm.keep_recent_turns
            and [(t.user, t.assistant) for t in loaded.turns]
            == [(t.user, t.assistant) for t in cm.turns]
            and loaded.num_compressions == cm.num_compressions
            and loaded.turns_folded == cm.turns_folded
        )
        ok &= cond
        print(f"  {PASS if cond else FAIL} load 完整还原了会话状态")

        # SessionStore 多会话
        store = SessionStore(os.path.join(tmp, "sessions"))
        store.save("alice", cm)
        cm2 = ContextManager(tok, system_prompt="bob 的会话")
        cm2.add_user("hi")
        cm2.add_assistant("hello")
        store.save("bob", cm2)

        listed = store.list_sessions()
        cond = listed == ["alice", "bob"]
        ok &= cond
        print(f"  {PASS if cond else FAIL} SessionStore 列出多个会话 {listed}")

        back = store.load("bob", tok)
        cond = back.system_prompt == "bob 的会话" and back.turns[0].user == "hi"
        ok &= cond
        print(f"  {PASS if cond else FAIL} SessionStore 读回指定会话")

    print(f"  Result: {'all passed' if ok else 'FAILED'}")
    return ok


# ===========================================================================
# Stage 4: e2e（真实引擎）—— 慢，需要下载模型
# ===========================================================================
def stage_e2e() -> bool:
    _header("Phase 5 – 端到端：真实引擎多轮对话 + 压缩")
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from tiny_inference import GenerationConfig, TinyQwenEngine

    model = os.environ.get("PHASE5_MODEL", "Qwen/Qwen3.5-0.8B")
    print(f"  加载引擎 {model} ...（首次会下载权重）")
    engine = TinyQwenEngine(model)  # prefix cache 默认关闭，隔离 Phase 5 自身评测

    def summarize_fn(prompt: str) -> str:
        out = engine.generate(
            messages=[{"role": "user", "content": prompt}],
            gen_config=GenerationConfig(max_new_tokens=80, do_sample=False),
        )
        return out["text"]

    cm = ContextManager(
        engine.tokenizer,
        system_prompt="You are a concise, helpful assistant.",
        max_context_tokens=320,   # 故意调小，逼出压缩
        reserve_for_reply=96,
        keep_recent_turns=2,
    )

    prompts = [
        "Hi! Please remember that my favorite color is teal.",
        "What is the capital of France?",
        "Name one programming language.",
        "What is 2 + 2?",
        "By the way, what is my favorite color?",
    ]
    gen = GenerationConfig(max_new_tokens=64, do_sample=False)
    for i, p in enumerate(prompts):
        cm.add_user(p)
        compressed = cm.maybe_compress(summarize_fn)
        msgs = cm.build_messages()
        reply = engine.generate(messages=msgs, gen_config=gen)["text"]
        cm.add_assistant(reply)
        tag = " [compressed]" if compressed else ""
        print(f"  turn {i}{tag}: tokens={cm.count_tokens(cm.build_messages())} reply={reply[:60]!r}")

    ok = True
    cond = cm.num_compressions >= 1
    ok &= cond
    print(f"  {PASS if cond else FAIL} 过程中触发过压缩 (num_compressions={cm.num_compressions})")

    cond = len(cm.turns) <= cm.keep_recent_turns or cm.count_tokens(cm.build_messages()) <= cm.token_budget()
    ok &= cond
    print(f"  {PASS if cond else FAIL} 末轮仍在预算内")

    # 信息性：摘要是否保住了「最喜欢的颜色」这个早期事实（模型能力相关，不强制断言）
    last = cm.turns[-1].assistant.lower()
    print(f"  [info] 末轮回复是否记得 teal: {'teal' in last}")

    print(f"  Result: {'all passed' if ok else 'FAILED'}")
    return ok


STAGES = {
    "compression": stage_compression,
    "prefix": stage_prefix,
    "persistence": stage_persistence,
    "e2e": stage_e2e,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 grader")
    parser.add_argument(
        "--stage",
        default="core",
        choices=["core", "all", *STAGES.keys()],
        help="core=三个逻辑测试(默认)，all=含 e2e，或指定单个 stage。",
    )
    args = parser.parse_args()

    if args.stage == "core":
        names = ["compression", "prefix", "persistence"]
    elif args.stage == "all":
        names = list(STAGES.keys())
    else:
        names = [args.stage]

    results = {name: STAGES[name]() for name in names}
    print("=" * 60)
    total = sum(results.values())
    for name, passed in results.items():
        print(f"  {name:12s}: {'PASS' if passed else 'FAIL'}")
    print(f"  Overall: {total}/{len(results)} stages passed")
    print("=" * 60)
    sys.exit(0 if total == len(results) else 1)


if __name__ == "__main__":
    main()
