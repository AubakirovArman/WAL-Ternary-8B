"""Command-line interface for WAL-Ternary direct-packed inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from .generation import greedy_generate, prompt_ids
from .cache import build_hardware_cache
from .loader import (
    checkpoint_runtime, choose_device, inspect_checkpoint, load_model,
    resolve_checkpoint,
)
from .platform import detect_platform


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@torch.inference_mode()
def _benchmark_cuda_graph(runner, ids: torch.Tensor, tokens: int) -> tuple[dict, dict]:
    correctness = runner.generate_ids(ids, max_new_tokens=min(16, tokens))
    torch.cuda.synchronize()
    output = runner._prefill(ids)
    runner.static_token.copy_(output.logits[:, -1].argmax(-1, keepdim=True))
    runner.static_position.fill_(ids.shape[1])
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.monotonic()
    begin.record()
    for _ in range(tokens):
        runner.graph.replay()
        runner.static_token.copy_(runner.static_next)
        runner.static_position.add_(1)
    end.record()
    end.synchronize()
    gpu_seconds = begin.elapsed_time(end) / 1000.0
    wall_seconds = time.monotonic() - wall_started
    return correctness, {
        "tokens": tokens, "gpu_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "gpu_tokens_per_second": tokens / gpu_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
    }


def _common_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="local checkpoint directory or Hugging Face model id")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0")
    parser.add_argument(
        "--backend", default="auto",
        choices=("auto", "portable", "cpu-v4", "cuda-v18"),
    )
    parser.add_argument("--hardware-cache", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wal-runtime")
    root.add_argument("--version", action="version", version="wal-runtime 0.2.0")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show available hardware and build tools")
    inspect = commands.add_parser("inspect", help="inspect a WAL checkpoint")
    inspect.add_argument("model")
    prepare = commands.add_parser(
        "prepare", help="build the portable direct-packed hardware cache",
    )
    prepare.add_argument("model")
    prepare.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate", help="generate one greedy response")
    _common_model(generate)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument(
        "--max-cache-len", type=int, default=0,
        help="static KV length; 0 chooses the smallest power of two that fits",
    )
    generate.add_argument("--threads", type=int, default=0)
    generate.add_argument("--repeat-tail-guard", type=int, default=16)
    benchmark = commands.add_parser("benchmark", help="single-stream decode benchmark")
    _common_model(benchmark)
    benchmark.add_argument("--prompt", default="In one sentence, why is the sky blue?")
    benchmark.add_argument("--tokens", type=int, default=32)
    benchmark.add_argument("--max-cache-len", type=int, default=0)
    benchmark.add_argument("--threads", type=int, default=0)
    return root


def _run_generation(args, *, emit_text: bool) -> int:
    checkpoint = resolve_checkpoint(args.model)
    if args.backend == "cpu-v4":
        if args.device not in {"auto", "cpu"}:
            raise ValueError("cpu-v4 requires --device cpu")
        device = "cpu"
    elif args.backend == "cuda-v18":
        if args.device not in {"auto", "cuda", "cuda:0"}:
            raise ValueError("cuda-v18 requires a CUDA device")
        device = "cuda:0" if args.device in {"auto", "cuda"} else args.device
    else:
        device = choose_device(args.device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    threads = args.threads or detect_platform().recommended_cpu_threads
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        torch.set_num_threads(threads)
    cache = args.hardware_cache.resolve() if args.hardware_cache else None
    model, report, backend = load_model(
        checkpoint, device=device, hardware_cache=cache, backend=args.backend,
        verify_hashes=not args.skip_hashes,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    ids = prompt_ids(tokenizer, args.prompt, device)
    required_cache = int(ids.shape[1]) + int(args.max_new_tokens) + 1
    max_cache_len = args.max_cache_len or max(
        256, 1 << (required_cache - 1).bit_length(),
    )
    if max_cache_len < required_cache:
        raise ValueError(
            f"--max-cache-len {max_cache_len} is smaller than required {required_cache}"
        )
    # CUDA Graph is the accepted V18 single-stream path.
    if backend == "cuda-v18":
        from .loader import development_module
        graph = development_module("qwen_v77_cuda_graph_greedy_v7")
        runner = graph.V7GraphGreedy(
            model, tokenizer, max_cache_len=max_cache_len,
        )
        if args.command == "benchmark":
            # First call compiles lazy extensions and captures the graph.  It
            # is deliberately excluded from steady-state decode throughput.
            correctness, timing = _benchmark_cuda_graph(
                runner, ids, args.max_new_tokens,
            )
            token_ids = correctness["all_token_ids"]
            result = {
                "token_ids": token_ids,
                "text": tokenizer.decode(
                    correctness["content_token_ids"], skip_special_tokens=True,
                ),
                "correctness_tokens": len(token_ids),
                **timing,
            }
        else:
            started = time.monotonic()
            result = runner.generate_ids(
                ids, max_new_tokens=args.max_new_tokens,
                repeat_tail_guard=getattr(args, "repeat_tail_guard", 16),
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            result["token_ids"] = result.pop("all_token_ids")
            result["text"] = tokenizer.decode(
                result.pop("content_token_ids"), skip_special_tokens=True,
            )
            result["seconds"] = elapsed
            result["tokens"] = len(result["token_ids"])
            result["tokens_per_second"] = (
                result["tokens"] / elapsed if elapsed else None
            )
    else:
        result = greedy_generate(
            model, tokenizer, ids, max_new_tokens=args.max_new_tokens,
            eos_ids=(tokenizer.eos_token_id, 151643),
            repeat_tail_guard=getattr(args, "repeat_tail_guard", 16),
        )
    payload = {
        "status": "ok", "backend": backend, "device": device,
        "threads": threads if device == "cpu" else None,
        "max_cache_len": max_cache_len,
        "model": str(checkpoint), "prompt_tokens": int(ids.numel()),
        "generation": result, "load_report": report,
    }
    if device.startswith("cuda"):
        payload["memory"] = {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    if emit_text:
        print(result["text"])
        print(json.dumps({key: payload[key] for key in (
            "backend", "device", "prompt_tokens"
        )}, ensure_ascii=False), file=sys.stderr)
    else:
        _json(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "doctor":
        info = detect_platform().as_dict()
        info["recommended_device"] = choose_device("auto")
        info["apple_silicon_status"] = (
            "portable MPS/CPU path available; native NEON/Metal optimization pending hardware validation"
            if info["apple_silicon"] else "not running on Apple Silicon"
        )
        _json(info)
        return 0
    if args.command == "inspect":
        _json(inspect_checkpoint(resolve_checkpoint(args.model)))
        return 0
    if args.command == "prepare":
        checkpoint = resolve_checkpoint(args.model)
        def progress(done, total, row):
            print(json.dumps({
                "event": "matrix", "completed": done, "total": total,
                "name": row["name"], "bytes": row["bytes"],
            }), flush=True)
        report = build_hardware_cache(
            checkpoint, args.output, checkpoint_runtime(checkpoint),
            progress=progress,
        )
        _json({key: report[key] for key in (
            "schema", "matrix_count", "payload_bytes", "seconds",
        )})
        return 0
    if args.command == "generate":
        return _run_generation(args, emit_text=True)
    if args.command == "benchmark":
        args.max_new_tokens = args.tokens
        args.repeat_tail_guard = 0
        return _run_generation(args, emit_text=False)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
