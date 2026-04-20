"""
Phase 4 correctness / persistence / speed test for SSD-offloaded Prefix Cache.

Usage:
    python test_phase4.py
    python test_phase4.py --stage correctness
    python test_phase4.py --stage persistence
    python test_phase4.py --stage speed

Design
------
- Correctness + Eviction: run 3 distinct long prompts with mem capacity = 1.
  After 3 inserts, 2 entries must have been offloaded to SSD. Re-querying one
  of the evicted prompts should report ssd_hits > 0 and still produce a
  correct answer keyword.
- Persistence: write some entries with engine A, DELETE engine A, then create
  a fresh engine B pointing at the SAME ssd directory. B's first lookup on
  a previously-seen prompt must hit the SSD tier (proving index was rebuilt
  from disk). This is the whole point of phase 4.
- Speed: compare prefill latency of (mem hit) vs (ssd hit) vs (no cache) on
  the same long prompt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tiny_inference import GenerationConfig, TinyQwenEngine  # noqa: E402

GREEDY = GenerationConfig(max_new_tokens=32, temperature=0, do_sample=False)
SPEED = GenerationConfig(max_new_tokens=16, temperature=0, do_sample=False)


LONG_SYSTEM = (
    "You are a concise assistant. "
    "Answer every question in as few words as possible. "
    "Never add greetings, disclaimers, or filler. "
    "Output only the final answer — no explanation, no punctuation unless necessary. "
) * 4


def build_messages(follow_up: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": LONG_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": follow_up}]},
    ]


TEST_CASES: list[tuple[str, list[str], str]] = [
    ("What is the capital of France? Answer in one word.", ["paris"], "Capital of France"),
    ("How many days are in a week? Answer with just the number.", ["7", "seven"], "Days in a week"),
    ("What color is the sky on a clear day? Answer in one word.", ["blue"], "Sky is blue"),
]


# ── Correctness + Eviction ───────────────────────────────────────────────────

def run_correctness(model: str, ssd_dir: str) -> int:
    print("\n" + "=" * 60)
    print("  Phase 4 SSD Offload – Correctness + Eviction")
    print("=" * 60)

    # 内存层容量故意设为 1：每插入第 2 条，第 1 条必被 offload 到 SSD。
    print(f"Loading model: {model}")
    engine = TinyQwenEngine(
        model,
        enable_prefix_cache=True,
        prefix_cache_ssd_dir=ssd_dir,
        prefix_cache_mem_entries=1,
    )

    failed = 0

    # Turn 1..3: 每轮用不同 prompt，一次 miss + 一次 hit（立即命中内存）
    for idx, (q, _kws, label) in enumerate(TEST_CASES):
        messages = build_messages(q)
        try:
            _ = engine.generate(messages=messages, gen_config=GREEDY)  # miss → insert
            second = engine.generate(messages=messages, gen_config=GREEDY, benchmark=True)
        except NotImplementedError as e:
            print(f"  [SKIP] {label}: {e}")
            failed += 1
            continue

        hit = second.get("metrics", {}).get("prefix_hit_tokens", 0)
        ok = hit > 0
        pc = engine.prefix_cache
        mem_n = len(pc._mem) if hasattr(pc, "_mem") else 0
        ssd_n = len(pc._disk) if hasattr(pc, "_disk") else 0
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:24s}  (mem={mem_n}, ssd={ssd_n}, hit_tokens={hit})")
        if not ok:
            failed += 1

    # 回头查 TEST_CASES[0]：它在 turn 2、turn 3 时已被 offload 到 SSD，应命中 SSD。
    q0, kws0, label0 = TEST_CASES[0]
    ssd_hits_before = getattr(engine.prefix_cache, "ssd_hits", 0)
    try:
        r = engine.generate(messages=build_messages(q0), gen_config=GREEDY, benchmark=True)
        text = r.get("text", "").strip()
    except NotImplementedError as e:
        print(f"  [SKIP] Re-query evicted: {e}")
        return failed + 1

    ssd_hits_after = getattr(engine.prefix_cache, "ssd_hits", 0)
    gained_ssd_hit = ssd_hits_after > ssd_hits_before
    has_keyword = any(k.lower() in text.lower() for k in kws0)
    hit_tokens = r.get("metrics", {}).get("prefix_hit_tokens", 0)
    ok = gained_ssd_hit and has_keyword and hit_tokens > 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Re-query evicted          "
          f"(ssd_hits gained={gained_ssd_hit}, hit_tokens={hit_tokens}, kw_ok={has_keyword})")
    if not ok:
        failed += 1
        print(f"         output: {text!r}")

    total = len(TEST_CASES) + 1
    print("-" * 60)
    print(f"  Result: {total - failed}/{total} passed")
    print("=" * 60)
    return failed


# ── Cross-process persistence ────────────────────────────────────────────────

def run_persistence(model: str, ssd_dir: str) -> int:
    print("\n" + "=" * 60)
    print("  Phase 4 SSD Offload – Cross-Process Persistence")
    print("=" * 60)

    # Writer engine：把若干条快照写到 SSD
    print(f"Writer: loading model {model}")
    writer = TinyQwenEngine(
        model,
        enable_prefix_cache=True,
        prefix_cache_ssd_dir=ssd_dir,
        prefix_cache_mem_entries=1,  # 小容量 → 强制 offload
    )
    try:
        for q, _, _ in TEST_CASES[:2]:
            writer.generate(messages=build_messages(q), gen_config=GREEDY)
    except NotImplementedError as e:
        print(f"  [SKIP] {e}")
        return 1

    on_disk = len(writer.prefix_cache._disk)
    print(f"  Writer engine finished, entries on disk: {on_disk}")
    del writer  # 模拟进程退出

    # Reader engine：全新实例，同一 ssd_dir → 期望启动时扫盘恢复索引
    print("Reader: creating fresh engine on same ssd_dir")
    reader = TinyQwenEngine(
        model,
        enable_prefix_cache=True,
        prefix_cache_ssd_dir=ssd_dir,
        prefix_cache_mem_entries=1,
    )
    idx_size = len(reader.prefix_cache._disk)
    print(f"  Rebuilt SSD index size: {idx_size}")
    if idx_size < on_disk:
        print("  [FAIL] 扫盘后索引数量少于 writer 写入，DiskStore._rebuild_index 可能没实现。")
        return 1

    # 请求一个曾被 writer 处理过的 prompt
    q, kws, _ = TEST_CASES[0]
    try:
        r = reader.generate(messages=build_messages(q), gen_config=GREEDY, benchmark=True)
    except NotImplementedError as e:
        print(f"  [SKIP] {e}")
        return 1

    ssd_hits = getattr(reader.prefix_cache, "ssd_hits", 0)
    hit_tokens = r.get("metrics", {}).get("prefix_hit_tokens", 0)
    kw_ok = any(k.lower() in r.get("text", "").lower() for k in kws)
    ok = ssd_hits > 0 and hit_tokens > 0 and kw_ok

    status = "[OK]" if ok else "[FAIL]"
    print(f"  Fresh engine lookup → ssd_hits={ssd_hits}, hit_tokens={hit_tokens}  {status}")
    print("=" * 60)
    return 0 if ok else 1


# ── Speed ────────────────────────────────────────────────────────────────────

def run_speed(model: str, ssd_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Phase 4 SSD Offload – Speed Comparison")
    print("=" * 60)

    print(f"Loading model: {model}")
    engine_no = TinyQwenEngine(model, enable_prefix_cache=False)
    engine_pc = TinyQwenEngine(
        model,
        enable_prefix_cache=True,
        prefix_cache_ssd_dir=ssd_dir,
        prefix_cache_mem_entries=1,
    )

    q = "Capital of Japan?"
    filler = "Name three primary colors."
    messages = build_messages(q)

    try:
        # 预热 no-cache 基线
        engine_no.generate(messages=messages, gen_config=SPEED)
        r_no = engine_no.generate(messages=messages, gen_config=SPEED, benchmark=True)

        # 写入 & 命中内存
        engine_pc.generate(messages=messages, gen_config=SPEED)          # miss → insert (mem)
        r_mem = engine_pc.generate(messages=messages, gen_config=SPEED, benchmark=True)  # mem hit

        # 让其被 offload：插入另一条不同 prompt（内存只容纳 1 条）
        engine_pc.generate(messages=build_messages(filler), gen_config=SPEED)

        # 再请求第一条 → 应命中 SSD
        r_ssd = engine_pc.generate(messages=messages, gen_config=SPEED, benchmark=True)
    except NotImplementedError as e:
        print(f"  [SKIP] {e}")
        return

    p_no = r_no["metrics"]["prefill_s"]
    p_mem = r_mem["metrics"]["prefill_s"]
    p_ssd = r_ssd["metrics"]["prefill_s"]
    hit_mem = r_mem["metrics"].get("prefix_hit_tokens", 0)
    hit_ssd = r_ssd["metrics"].get("prefix_hit_tokens", 0)

    print(f"  Prompt length: {r_no['usage']['prompt_tokens']} tokens")
    print(f"  mem hit_tokens = {hit_mem}, ssd hit_tokens = {hit_ssd}")
    print()
    print(f"  {'':24s} {'mem hit':>10s} {'ssd hit':>10s} {'no cache':>10s}")
    print(f"  {'Prefill elapsed (s)':24s} {p_mem:>10.4f} {p_ssd:>10.4f} {p_no:>10.4f}")
    print()
    print(f"  mem-hit speedup vs no-cache : {p_no / max(p_mem, 1e-6):.2f}×")
    print(f"  ssd-hit speedup vs no-cache : {p_no / max(p_ssd, 1e-6):.2f}×")
    if hit_mem > 0 and hit_ssd > 0 and p_mem < p_no and p_ssd < p_no:
        print("  [OK] SSD 命中确实省掉了绝大部分 prefill（虽然比内存命中慢一些）。")
    else:
        print("  [WARN] 看起来有一环没生效，检查 TieredPrefixCache 的 lookup/insert / DiskStore 实现。")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 SSD-Offloaded Prefix Cache test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--stage",
        choices=["correctness", "persistence", "speed"],
        default=None,
        help="Run only one stage. Default: run all.",
    )
    parser.add_argument(
        "--ssd-dir",
        default=None,
        help="SSD cache directory. Default: a fresh temp dir per stage.",
    )
    args = parser.parse_args()

    failed = 0
    stages = [args.stage] if args.stage else ["correctness", "persistence", "speed"]

    for stage in stages:
        # 每个 stage 用独立目录，避免互相污染
        ssd_dir = args.ssd_dir or tempfile.mkdtemp(prefix=f"phase4_{stage}_")
        try:
            if stage == "correctness":
                failed += run_correctness(args.model, ssd_dir)
            elif stage == "persistence":
                failed += run_persistence(args.model, ssd_dir)
            elif stage == "speed":
                run_speed(args.model, ssd_dir)
        finally:
            if args.ssd_dir is None and os.path.isdir(ssd_dir):
                shutil.rmtree(ssd_dir, ignore_errors=True)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
