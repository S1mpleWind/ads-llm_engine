"""
Phase 2 correctness test.

Usage:
    python test_phase2.py
    python test_phase2.py --model Qwen/Qwen3.5-0.8B
    python test_phase2.py --speed   # also run a prefill/decode speed comparison

Each test case sends a simple factual question to the model (temperature=0, greedy
decoding) and checks that the expected keyword appears in the output.  A working
KV Cache implementation produces coherent answers; a broken one typically produces
repetition, garbage, or the wrong answer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── mirror for mainland China ─────────────────────────────────────────────────
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tiny_inference import GenerationConfig, TinyQwenEngine  # noqa: E402

# ── greedy, deterministic config ─────────────────────────────────────────────
GREEDY = GenerationConfig(
    max_new_tokens=64,
    temperature=0,
    do_sample=False,
)

# ── test cases ────────────────────────────────────────────────────────────────
# Each entry: (question, [accepted keywords], description)
# At least ONE keyword must appear (case-insensitive) for the test to pass.
TEST_CASES: list[tuple[str, list[str], str]] = [
    (
        "What is 1 + 1? Answer with just the number.",
        ["2", "two"],
        "1+1=2",
    ),
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
    (
        "What is 3 multiplied by 4? Answer with just the number.",
        ["12", "twelve"],
        "3×4=12",
    ),
    (
        "Is water wet? Answer Yes or No.",
        ["yes"],
        "Water is wet",
    ),
]


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def run_tests(engine: TinyQwenEngine) -> tuple[int, int]:
    passed = 0
    failed = 0

    use_cache = True

    print("\n" + "=" * 60)
    print("  Phase 2 KV Cache – Correctness Tests")
    print("=" * 60)

    for question, keywords, label in TEST_CASES:
        try:
            result = engine.generate(
                messages=build_messages(question),
                gen_config=GREEDY,
                use_cache=use_cache,
            )
        except NotImplementedError:
            if use_cache:
                print("  [NOTE] KV Cache 尚未实现，自动切换到 no-cache 模式继续测试")
                use_cache = False
            result = engine.generate(
                messages=build_messages(question),
                gen_config=GREEDY,
                use_cache=False,
            )
        answer = result["text"].strip().lower()

        ok = any(kw.lower() in answer for kw in keywords)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {label}")
        if not ok:
            print(f"         question : {question}")
            print(f"         expected : {keywords}")
            print(f"         got      : {result['text'].strip()!r}")

    print("-" * 60)
    print(f"  Result: {passed}/{passed + failed} passed")
    print("=" * 60)

    return passed, failed


def run_speed_comparison(engine: TinyQwenEngine) -> None:
    """
    Compare KV Cache (phase2) vs no-cache (phase1 baseline) on one short prompt.
    With a correct KV Cache implementation, decode tokens/s should be noticeably
    faster (or at least not slower) than the no-cache baseline.
    """
    SPEED_CONFIG = GenerationConfig(max_new_tokens=32, temperature=0, do_sample=False)
    question = "Tell me a fun fact about the ocean."
    messages = build_messages(question)

    print("\n" + "=" * 60)
    print("  Phase 2 KV Cache – Speed Comparison")
    print("=" * 60)

    # ── with KV Cache ──────────────────────────────────────────────────────────
    try:
        r_cache = engine.generate(
            messages=messages,
            gen_config=SPEED_CONFIG,
            benchmark=True,
            use_cache=True,
        )
        m_cache = r_cache["metrics"]
        cache_implemented = True
    except NotImplementedError:
        m_cache = None
        cache_implemented = False

    # ── without KV Cache (Phase 1 baseline) ────────────────────────────────────
    r_nocache = engine.generate(
        messages=messages,
        gen_config=SPEED_CONFIG,
        benchmark=True,
        use_cache=False,
    )
    m_nocache = r_nocache["metrics"]

    def fmt(val):
        return f"{val:>12.2f}" if val is not None else f"{'NaN':>12}"

    def fmt3(val):
        return f"{val:>12.3f}" if val is not None else f"{'NaN':>12}"

    c_prefill = m_cache["prefill_tokens_per_s"] if cache_implemented else None
    c_decode  = m_cache["decode_tokens_per_s"]  if cache_implemented else None
    c_elapsed = m_cache["elapsed_s"]             if cache_implemented else None

    print(f"  {'':30s}  {'with cache':>12}  {'no cache':>12}")
    print(f"  {'Prefill (tokens/s)':30s}  {fmt(c_prefill)}  {m_nocache['prefill_tokens_per_s']:>12.2f}")
    print(f"  {'Decode  (tokens/s)':30s}  {fmt(c_decode)}  {m_nocache['decode_tokens_per_s']:>12.2f}")
    print(f"  {'Total elapsed (s)':30s}  {fmt3(c_elapsed)}  {m_nocache['elapsed_s']:>12.3f}")

    if not cache_implemented:
        print("\n  [WARN] with cache 列显示 NaN —— KV Cache decode 尚未实现，请完成 TODO 块。")
    else:
        ratio = m_cache["decode_tokens_per_s"] / max(m_nocache["decode_tokens_per_s"], 1e-6)
        print(f"\n  Decode speedup (cache / no-cache): {ratio:.2f}×")
        if ratio >= 1.0:
            print("  [OK] KV Cache decode is at least as fast as baseline.")
        else:
            print("  [WARN] KV Cache decode is slower than baseline — something may be wrong.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 KV Cache test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Model name or local path")
    parser.add_argument(
        "--stage",
        choices=["correctness", "speed"],
        default=None,
        help="Run only 'correctness' or only 'speed'. Default: run both.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    engine = TinyQwenEngine(args.model)

    failed = 0
    if args.stage in (None, "correctness"):
        _, failed = run_tests(engine)

    if args.stage in (None, "speed"):
        run_speed_comparison(engine)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
