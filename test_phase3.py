"""
Phase 3 correctness + speed test for Prefix Cache.

Usage:
    python test_phase3.py
    python test_phase3.py --model Qwen/Qwen3.5-0.8B
    python test_phase3.py --stage correctness
    python test_phase3.py --stage speed

Design
------
- Correctness: run the same question **twice** on an engine with Prefix Cache enabled.
  The second run should (a) hit the Prefix Cache (prefix_hit_tokens > 0) and (b) produce
  a semantically-correct answer (expected keyword appears). We do NOT require
  byte-exact equality with the no-prefix-cache baseline because Qwen3.5's linear
  attention runs different numerical paths in chunk-prefill vs. single-step recurrent
  decode; under greedy decoding tiny FP differences can flip tokens like "Blue" →
  "**Blue**". This is inherent to partial prefill on hybrid architectures.
- Speed: construct a **shared long system prompt** and two short follow-up user
  turns. The second turn shares the whole history-so-far with the first turn as
  its prefix, so Prefix Cache should eliminate almost all of its prefill cost.
"""
from __future__ import annotations

import argparse
import os
import sys

# ── mirror for mainland China ─────────────────────────────────────────────────
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tiny_inference import GenerationConfig, TinyQwenEngine  # noqa: E402

GREEDY = GenerationConfig(
    max_new_tokens=32,
    temperature=0,
    do_sample=False,
)

TEST_CASES: list[tuple[str, list[str], str]] = [
    (
        "What is the capital of France? Answer in one word.",
        ["paris"],
        "Capital of France",
    ),
    (
        "How many days are in a week? Answer with just the number.",
        ["7", "seven"],
        "Days in a week",
    ),
    (
        "What color is the sky on a clear day? Answer in one word.",
        ["blue"],
        "Sky is blue",
    ),
]


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


# ── Correctness ──────────────────────────────────────────────────────────────

def run_correctness(model: str) -> int:
    print("\n" + "=" * 60)
    print("  Phase 3 Prefix Cache – Correctness Tests")
    print("=" * 60)

    print(f"Loading model: {model}")
    engine_nocache = TinyQwenEngine(model, enable_prefix_cache=False)
    engine_prefix = TinyQwenEngine(model, enable_prefix_cache=True)

    failed = 0
    for question, keywords, label in TEST_CASES:
        messages = build_messages(question)
        try:
            baseline = engine_nocache.generate(messages=messages, gen_config=GREEDY).get("text", "").strip()
            # First call with Prefix Cache: should populate the cache (miss).
            first = engine_prefix.generate(messages=messages, gen_config=GREEDY).get("text", "").strip()
            # Second call with same messages: should be a prefix-cache hit.
            second = engine_prefix.generate(messages=messages, gen_config=GREEDY, benchmark=True)
            hit_tokens = second.get("metrics", {}).get("prefix_hit_tokens", 0)
            second_text = second.get("text", "").strip()
        except NotImplementedError as e:
            print(f"  [SKIP] {label}: {e}")
            failed += 1
            continue

        has_keyword = any(k.lower() in second_text.lower() for k in keywords)
        prefix_hit = hit_tokens > 0

        ok = has_keyword and prefix_hit
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {label}  (prefix_hit_tokens={hit_tokens})")
        if not ok:
            print(f"         baseline: {baseline!r}")
            print(f"         1st run : {first!r}")
            print(f"         2nd run : {second_text!r}")

    passed = len(TEST_CASES) - failed
    print("-" * 60)
    print(f"  Result: {passed}/{len(TEST_CASES)} passed")
    print("=" * 60)
    return failed


# ── Speed ────────────────────────────────────────────────────────────────────

LONG_SYSTEM = (
    "You are a concise assistant. "
    "Answer every question in as few words as possible. "
    "Never add greetings, disclaimers, or filler. "
    "Output only the final answer — no explanation, no punctuation unless necessary. "
) * 4  # intentionally long to make prefill dominant


def build_long_messages(follow_up: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": LONG_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": follow_up}]},
    ]


def run_speed(model: str) -> None:
    print("\n" + "=" * 60)
    print("  Phase 3 Prefix Cache – Speed Comparison")
    print("=" * 60)

    SPEED_CONFIG = GenerationConfig(max_new_tokens=16, temperature=0, do_sample=False)

    print(f"Loading model: {model}")
    # Two engines: one with Prefix Cache, one without. Same prompt sent twice.
    engine_no = TinyQwenEngine(model, enable_prefix_cache=False)
    engine_pc = TinyQwenEngine(model, enable_prefix_cache=True)

    question = "Capital of Japan?"
    messages = build_long_messages(question)

    try:
        # Warm up both engines with the same prompt so that the second call is the measured one.
        r_no_1 = engine_no.generate(messages=messages, gen_config=SPEED_CONFIG, benchmark=True)
        r_pc_1 = engine_pc.generate(messages=messages, gen_config=SPEED_CONFIG, benchmark=True)

        # Measured second call (same messages → full prefix hit for the prefix-cache engine).
        r_no_2 = engine_no.generate(messages=messages, gen_config=SPEED_CONFIG, benchmark=True)
        r_pc_2 = engine_pc.generate(messages=messages, gen_config=SPEED_CONFIG, benchmark=True)
    except NotImplementedError as e:
        print(f"  [SKIP] Prefix Cache 尚未实现：{e}")
        return

    m_no = r_no_2["metrics"]
    m_pc = r_pc_2["metrics"]
    hit = m_pc.get("prefix_hit_tokens", 0)

    print(f"  Prompt length (tokens): {r_no_2['usage']['prompt_tokens']}")
    print(f"  Prefix hit tokens (second call, Prefix Cache): {hit}")
    print()
    print(f"  {'':30s}  {'w/ Prefix Cache':>16}  {'no Prefix Cache':>16}")
    print(f"  {'Prefill elapsed (s)':30s}  {m_pc['prefill_s']:>16.4f}  {m_no['prefill_s']:>16.4f}")
    print(f"  {'Total elapsed (s)':30s}  {m_pc['elapsed_s']:>16.4f}  {m_no['elapsed_s']:>16.4f}")

    speedup = m_no["prefill_s"] / max(m_pc["prefill_s"], 1e-6)
    print(f"\n  Prefill speedup (second call, cache / no-cache): {speedup:.2f}×")
    if hit > 0 and speedup >= 1.5:
        print("  [OK] Prefix Cache 显著加速了第二次相同 prompt 的 prefill。")
    elif hit == 0:
        print("  [WARN] 第二次相同请求没有命中Prefix Cache，检查 lookup/insert 实现。")
    else:
        print("  [WARN] 命中了Prefix Cache但加速不明显，可能是 prompt 过短或 clone 开销占比过高。")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Prefix Cache test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Model name or local path")
    parser.add_argument(
        "--stage",
        choices=["correctness", "speed"],
        default=None,
        help="Run only 'correctness' or only 'speed'. Default: run both.",
    )
    args = parser.parse_args()

    failed = 0
    if args.stage in (None, "correctness"):
        failed = run_correctness(args.model)

    if args.stage in (None, "speed"):
        run_speed(args.model)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
