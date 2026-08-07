"""Greedy generation shared by CPU, MPS and CUDA CLI paths."""
from __future__ import annotations

from collections.abc import Sequence
import time

import torch


def prompt_ids(tokenizer, prompt: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )
    except (TypeError, ValueError):
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
    return ids.to(device)


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    repeat_tail_guard: int | None = 16,
    repeat_search_window: int = 80,
) -> dict:
    started = time.monotonic()
    prefill_started = time.monotonic()
    state = model(input_ids, use_cache=True)
    prefill_seconds = time.monotonic() - prefill_started
    past = state.past_key_values
    token = state.logits[:, -1].argmax(-1, keepdim=True)
    generated: list[int] = []
    stopped_by = None
    eos = {int(value) for value in eos_ids if value is not None}
    decode_started = time.monotonic()
    decode_steps = 0
    for step in range(max_new_tokens):
        value = int(token.item())
        generated.append(value)
        if value in eos:
            stopped_by = "eos"
            break
        if repeat_tail_guard and len(generated) >= 2 * repeat_tail_guard:
            tail = generated[-repeat_tail_guard:]
            begin = max(0, len(generated) - repeat_search_window)
            if any(
                generated[index:index + repeat_tail_guard] == tail
                for index in range(begin, len(generated) - repeat_tail_guard)
            ):
                del generated[-repeat_tail_guard:]
                stopped_by = "repeat_tail_guard"
                break
        if step + 1 == max_new_tokens:
            break
        state = model(token, past_key_values=past, use_cache=True)
        decode_steps += 1
        past = state.past_key_values
        token = state.logits[:, -1].argmax(-1, keepdim=True)
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    decode_seconds = time.monotonic() - decode_started
    elapsed = time.monotonic() - started
    content = generated[:-1] if generated and generated[-1] in eos else generated
    return {
        "token_ids": generated,
        "text": tokenizer.decode(content, skip_special_tokens=True),
        "tokens": len(generated),
        "seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed if elapsed else None,
        "prefill_seconds": prefill_seconds,
        "decode_steps": decode_steps,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            decode_steps / decode_seconds if decode_seconds else None
        ),
        "stopped_by": stopped_by,
    }
