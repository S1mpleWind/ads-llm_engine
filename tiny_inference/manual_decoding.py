from __future__ import annotations

import time

import torch

from .config import GenerationConfig
from .manual_qwen3_5 import qwen3_5_text_forward
from .sampling import sample_next_token


def _eos_token_ids(model) -> set[int]:
    """Match HF GenerationMixin: stop if *any* configured EOS id is sampled."""
    raw = getattr(model.config, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return {int(x) for x in raw if x is not None}
    return {int(raw)}


@torch.no_grad()
def decode_tokens_manual(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
) -> tuple[list[int], torch.Tensor, dict[str, float]]:
    model.eval()
    generated: list[int] = []
    eos_ids = _eos_token_ids(model)


    prefill_start = time.time()
    logits, past_key_values = qwen3_5_text_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=use_cache,
    )
    prefill_end = time.time()

    # 从 prefill 输出的最后一个位置采样第一个新 token
    logits = logits[:, -1, :]
    next_token = sample_next_token(logits, gen_config)
    token_id = int(next_token.item())
    generated.append(token_id)

    attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token)], dim=1
    )

    decode_start = time.time()
    for step in range(1, gen_config.max_new_tokens):
        if token_id in eos_ids:
            break

        if use_cache:
            # ===== TODO: KV Cache - Decode 单步（用缓存替代全序列重计算）(START) =====
            # 调用 qwen3_5_text_forward，只传入本步新生成的 1 个 token（而非完整序列），
            # 同时传入 past_key_values（已有缓存）和 use_cache=True。
            # 函数返回新的 logits 和更新后的 past_key_values，供下一步继续使用。
            raise NotImplementedError("KV Cache decode 尚未实现，请完成此 TODO 块")
            
            # ===== TODO: KV Cache - Decode 单步（用缓存替代全序列重计算）(END) =====
        else:
            full_ids = torch.cat(
                [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
            )
            logits, _ = qwen3_5_text_forward(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
            )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        token_id = int(next_token.item())
        generated.append(token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
    decode_end = time.time()

    full_ids = torch.cat(
        [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
    )

    timing = {
        "prefill_s": prefill_end - prefill_start,
        "decode_s": decode_end - decode_start,
        "decode_tokens": max(len(generated) - 1, 1),
        "prompt_tokens": int(input_ids.shape[1]),
    }
    return generated, full_ids, timing


@torch.no_grad()
def decode_stream_manual(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
):
    model.eval()
    eos_ids = _eos_token_ids(model)

    # 将完整 prompt 一次性喂入模型，内部自动创建并填充所有层的缓存
    prefill_start = time.time()
    logits, past_key_values = qwen3_5_text_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=use_cache,
    )
    prefill_end = time.time()

    logits = logits[:, -1, :]
    next_token = sample_next_token(logits, gen_config)
    token_id = int(next_token.item())
    generated: list[int] = [token_id]
    attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token)], dim=1
    )

    if token_id not in eos_ids:
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if piece:
            yield piece

    decode_start = time.time()
    decode_count = 0
    for step in range(1, gen_config.max_new_tokens):
        if token_id in eos_ids:
            break

        if use_cache:
            # ===== TODO: KV Cache - Streaming Decode 单步（用缓存替代全序列重计算）(START) =====
            # 启用缓存时：只输入刚生成的 1 个 token，携带已有缓存，forward 内部追加并返回更新后的缓存
            raise NotImplementedError("KV Cache decode 尚未实现，请完成此 TODO 块")
            
            # ===== TODO: KV Cache - Streaming Decode 单步（用缓存替代全序列重计算）(END) =====
        else:
            full_ids = torch.cat(
                [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
            )
            logits, _ = qwen3_5_text_forward(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
            )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        token_id = int(next_token.item())
        generated.append(token_id)
        decode_count += 1
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
        if token_id in eos_ids:
            break
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if piece:
            yield piece
    decode_end = time.time()

    # 输出 prefill / decode 速度统计
    prefill_time = prefill_end - prefill_start
    decode_time = decode_end - decode_start
    prompt_tokens = int(input_ids.shape[1])
    decode_tokens = max(decode_count, 1)

    yield f"\n\n--- Performance ---\n"
    yield f"Prefill : {prompt_tokens} tokens in {prefill_time:.3f}s → {prompt_tokens / max(prefill_time, 1e-6):.2f} tokens/s\n"
    yield f"Decode  : {decode_tokens} tokens in {decode_time:.3f}s → {decode_tokens / max(decode_time, 1e-6):.2f} tokens/s\n"
