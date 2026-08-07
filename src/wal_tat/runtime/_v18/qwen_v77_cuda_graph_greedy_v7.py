"""Reusable batch-one greedy generator for the direct-packed V7 CUDA runtime."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import StaticCache


class V7GraphGreedy:
    """Prefill arbitrary prompts and replay one captured V7 decode step."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_cache_len: int = 8192,
        eos_ids: Sequence[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_len = int(max_cache_len)
        configured = [tokenizer.eos_token_id, 151643] if eos_ids is None else eos_ids
        self.eos_ids = {int(token) for token in configured if token is not None}
        self.cache = StaticCache(model.config, max_cache_len=self.max_cache_len)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_token: torch.Tensor | None = None
        self.static_position: torch.Tensor | None = None
        self.static_next: torch.Tensor | None = None

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("V7 graph generator requires batch one")
        if input_ids.shape[1] >= self.max_cache_len:
            raise ValueError("prompt does not fit the static KV cache")
        self.cache.reset()
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long,
        )
        return self.model(
            input_ids, past_key_values=self.cache, use_cache=True,
            cache_position=positions,
        )

    @torch.inference_mode()
    def _capture(self, seed_ids: torch.Tensor) -> None:
        output = self._prefill(seed_ids)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        self.static_token = token.clone()
        self.static_position = torch.full(
            (1,), seed_ids.shape[1], device=seed_ids.device, dtype=torch.long,
        )
        # Compile all lazy kernels and establish stable allocator state.
        self.model(
            self.static_token, past_key_values=self.cache, use_cache=True,
            cache_position=self.static_position,
        )
        torch.cuda.synchronize()

        output = self._prefill(seed_ids)
        self.static_token.copy_(output.logits[:, -1].argmax(-1, keepdim=True))
        self.static_position.fill_(seed_ids.shape[1])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(
                self.static_token, past_key_values=self.cache, use_cache=True,
                cache_position=self.static_position,
            )
            self.static_next = static_output.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        self.graph = graph

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_strings: Sequence[str] = (),
        repeat_tail_guard: int | None = None,
        repeat_search_window: int = 80,
    ) -> dict:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("V7 graph generation requires batch one")
        if input_ids.shape[1] + max_new_tokens > self.max_cache_len:
            raise ValueError("prompt plus generation exceeds the static KV cache")
        if self.graph is None:
            self._capture(input_ids)
        assert self.graph is not None
        assert self.static_token is not None
        assert self.static_position is not None
        assert self.static_next is not None

        output = self._prefill(input_ids)
        first = output.logits[:, -1].argmax(-1, keepdim=True)
        self.static_token.copy_(first)
        self.static_position.fill_(input_ids.shape[1])
        torch.cuda.synchronize()

        generated: list[int] = []
        stopped_by: str | None = None

        def accept(token: int) -> bool:
            nonlocal stopped_by
            generated.append(token)
            if repeat_tail_guard:
                width = int(repeat_tail_guard)
                if width <= 0:
                    raise ValueError("repeat_tail_guard must be positive")
                if len(generated) >= 2 * width:
                    tail = generated[-width:]
                    start = max(0, len(generated) - int(repeat_search_window))
                    if any(
                        generated[index:index + width] == tail
                        for index in range(start, len(generated) - width)
                    ):
                        # Remove the first duplicated block.  This is exactly
                        # the output of the discovery-time postprocessor, but
                        # stops GPU replay immediately instead of computing the
                        # rest of a pathological tail.
                        del generated[-width:]
                        stopped_by = "repeat_tail_guard"
                        return True
            if token in self.eos_ids:
                stopped_by = "eos"
                return True
            if stop_strings:
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                if any(stop and stop in text for stop in stop_strings):
                    stopped_by = "stop_string"
                    return True
            return False

        if max_new_tokens > 0:
            if accept(int(first.item())):
                max_new_tokens = 1
        for _ in range(1, max_new_tokens):
            self.graph.replay()
            token = int(self.static_next.item())
            if accept(token):
                break
            self.static_token.copy_(self.static_next)
            self.static_position.add_(1)

        eos_position = next(
            (index for index, token in enumerate(generated) if token in self.eos_ids),
            None,
        )
        content = generated if eos_position is None else generated[:eos_position]
        return {
            "all_token_ids": generated,
            "content_token_ids": content,
            "eos": eos_position is not None,
            "stopped_by": stopped_by,
            "max_length_hit": stopped_by is None and len(generated) >= max_new_tokens,
        }
