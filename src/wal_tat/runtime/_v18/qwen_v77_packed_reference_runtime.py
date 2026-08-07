"""Correctness-first packed runtime for WAL-Ternary-8B V77.

The runtime never constructs a complete BF16/FP16 Transformer body.  It keeps
the checkpoint in its physical T3+sparse-k8 / WALB2 / INT3 / INT4 formats and
evaluates one output-row chunk at a time.  The current implementation decodes
logical codes on the CPU and performs the dot products with PyTorch on the
requested CPU or CUDA device.  It is intentionally a portable reference, not
the final fused performance kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping

import torch
from torch import nn


DEFAULT_CHECKPOINT = Path(
    "results/checkpoints/"
    "qwen3_t3_lb2_scaleaware_qo_kv_leaveout_v77_r12b1_lite"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bundled_runtime(checkpoint: str | Path) -> tuple[Any, Any]:
    root = Path(checkpoint).resolve(strict=True)
    runtime = root / "runtime"
    source = root / "src"
    for member in (runtime, source):
        if not member.is_dir():
            raise ValueError(f"missing bundled runtime directory: {member}")
    for member in (str(runtime), str(source)):
        if member not in sys.path:
            sys.path.insert(0, member)
    endpoint_name = f"wal_v77_endpoint_{abs(hash(str(root)))}"
    endpoint = sys.modules.get(endpoint_name)
    if endpoint is None:
        endpoint = _load_module(
            endpoint_name, runtime / "qwen_endpoint_lowbit_artifact.py"
        )
    import qwen_static_lowbit_artifact as static  # type: ignore
    endpoint._wal_static = static
    from wal_tat import binary_lowrank as codec  # type: ignore

    return endpoint, codec


def load_manifests(checkpoint: str | Path) -> tuple[Path, dict, Path, dict]:
    root = Path(checkpoint).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    base_manifest_path = root / str(manifest["base"]["manifest"])
    base_root = base_manifest_path.parent
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    return root, manifest, base_root, base_manifest


def _iter_t3_sparse_rows(
    endpoint: Any,
    path: Path,
    *,
    row_chunk: int,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield base codes, sparse codes, alpha and beta without dense weights."""

    header = endpoint._read_matrix_header(path)
    if header["spec"].name != "t3-sparse-k8":
        raise ValueError(f"unsupported body format: {header['spec'].name}")
    rows = int(header["rows"])
    columns = int(header["columns"])
    group_size = int(header["group_size"])
    groups_per_row = columns // group_size
    static = endpoint._wal_static
    code_bytes_per_row = groups_per_row * static.SPARSE_CODE_RECORD_BYTES
    metadata_bytes_per_row = groups_per_row * 2 * 2
    metadata_offset = endpoint.MATRIX_HEADER.size + int(header["code_bytes"])
    with path.open("rb") as handle:
        for start in range(0, rows, row_chunk):
            count = min(row_chunk, rows - start)
            handle.seek(endpoint.MATRIX_HEADER.size + start * code_bytes_per_row)
            code_raw = handle.read(count * code_bytes_per_row)
            handle.seek(metadata_offset + start * metadata_bytes_per_row)
            metadata_raw = handle.read(count * metadata_bytes_per_row)
            if len(code_raw) != count * code_bytes_per_row:
                raise ValueError("truncated T3+sparse-k8 codes")
            if len(metadata_raw) != count * metadata_bytes_per_row:
                raise ValueError("truncated T3+sparse-k8 metadata")
            packed = torch.frombuffer(bytearray(code_raw), dtype=torch.uint8)
            base, sparse = static._unpack_sparse_k8_records(packed)
            shape = (count, groups_per_row, group_size)
            base = base.reshape(shape)
            sparse = sparse.reshape(shape)
            metadata = torch.frombuffer(
                bytearray(metadata_raw), dtype=torch.float16
            ).reshape(count, groups_per_row, 2)
            if not torch.isfinite(metadata).all() or torch.any(metadata <= 0):
                raise ValueError("invalid T3+sparse-k8 metadata")
            yield start, base, sparse, metadata[..., 0], metadata[..., 1]


def _group_dot(
    value: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Compute grouped low-bit rows without constructing a dense weight."""

    flat = value.reshape(-1, value.shape[-1]).float()
    rows, groups, group_size = codes.shape
    if flat.shape[-1] != groups * group_size:
        raise ValueError("activation width does not match packed matrix")
    grouped = flat.reshape(flat.shape[0], groups, group_size)
    logical = codes.to(device=value.device, dtype=torch.float32)
    scale = scales.to(device=value.device, dtype=torch.float32)
    dots = torch.einsum("bgi,rgi->brg", grouped, logical)
    return (dots * scale.unsqueeze(0)).sum(-1).reshape(*value.shape[:-1], rows)


class PackedWALLinear(nn.Module):
    """T3+sparse-k8 base plus optional WALB2 correction."""

    def __init__(
        self,
        base_path: Path,
        *,
        endpoint: Any,
        codec: Any,
        overlay_path: Path | None,
        in_features: int,
        out_features: int,
        row_chunk: int = 64,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.overlay_path = None if overlay_path is None else Path(overlay_path)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.row_chunk = int(row_chunk)
        self.output_dtype = output_dtype
        self.endpoint = endpoint
        self.codec = codec
        self._overlay_cache: dict[str, tuple[Any, ...]] = {}

    def _overlay(self, device: torch.device) -> tuple[Any, ...]:
        if self.overlay_path is None:
            return ()
        key = str(device)
        cached = self._overlay_cache.get(key)
        if cached is not None:
            return cached
        bundle = self.codec.read_binary_lowrank_bundle(self.overlay_path)
        if tuple(bundle.shape) != (self.out_features, self.in_features):
            raise ValueError("WALB2 shape does not match T3 base")
        paths = []
        for item in bundle.paths:
            paths.append(
                (
                    item.u_codes.to(device=device, dtype=torch.int8),
                    item.v_codes.to(device=device, dtype=torch.int8),
                    item.row_scales_fp16.to(device=device),
                    item.latent_scales_fp16.to(device=device),
                    item.column_scales_fp16.to(device=device),
                )
            )
        cached = tuple(paths)
        self._overlay_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        chunks = []
        for _, base, sparse, alpha, beta in _iter_t3_sparse_rows(
            self.endpoint, self.base_path, row_chunk=self.row_chunk
        ):
            part = _group_dot(value, base, alpha)
            part.add_(_group_dot(value, sparse, beta))
            chunks.append(part)
        result = torch.cat(chunks, dim=-1)
        flat = value.reshape(-1, value.shape[-1]).float()
        correction = torch.zeros(
            (flat.shape[0], self.out_features),
            device=value.device,
            dtype=torch.float32,
        )
        for u_codes, v_codes, row, latent, column in self._overlay(value.device):
            hidden = (flat * column.float().unsqueeze(0)) @ v_codes.float().transpose(0, 1)
            hidden.mul_(latent.float().unsqueeze(0))
            left = u_codes.float() * row.float().unsqueeze(1)
            correction.add_(hidden @ left.transpose(0, 1))
        result.add_(correction.reshape(*value.shape[:-1], self.out_features))
        return result.to(dtype=self.output_dtype)


class PackedEndpointLinear(nn.Module):
    """Streaming INT3/INT4 output projection without a dense weight."""

    def __init__(
        self,
        path: Path,
        *,
        endpoint: Any,
        row_chunk: int,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.endpoint = endpoint
        header = endpoint._read_endpoint_header(self.path)
        payload = header["payload"]
        self.in_features = int(payload.columns)
        self.out_features = int(payload.rows)
        self.row_chunk = int(row_chunk)
        self.output_dtype = output_dtype

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        outputs = []
        for _, codes, scales in self.endpoint.iter_quantized_endpoint_rows(
            self.path, row_chunk=self.row_chunk
        ):
            outputs.append(_group_dot(value, codes, scales))
        return torch.cat(outputs, dim=-1).to(dtype=self.output_dtype)


class PackedEmbedding(nn.Module):
    """Random-access INT3 embedding lookup from the packed endpoint file."""

    def __init__(self, path: Path, *, endpoint: Any, output_dtype: torch.dtype) -> None:
        super().__init__()
        self.path = Path(path)
        self.endpoint = endpoint
        header = endpoint._read_endpoint_header(self.path)
        self.spec = header["spec"]
        self.payload = header["payload"]
        self.num_embeddings = int(self.payload.rows)
        self.embedding_dim = int(self.payload.columns)
        self.output_dtype = output_dtype

    def _rows(self, rows: list[int]) -> torch.Tensor:
        code_bytes = self.embedding_dim * int(self.spec.code_bits) // 8
        groups = self.embedding_dim // int(self.payload.group_size)
        metadata_bytes = groups * int(self.spec.scale_bits) // 8
        metadata_offset = (
            self.endpoint.ENDPOINT_HEADER.size + int(self.payload.code_bits) // 8
        )
        decoded = []
        with self.path.open("rb") as handle:
            for row in rows:
                handle.seek(self.endpoint.ENDPOINT_HEADER.size + row * code_bytes)
                raw_codes = handle.read(code_bytes)
                handle.seek(metadata_offset + row * metadata_bytes)
                raw_scales = handle.read(metadata_bytes)
                if len(raw_codes) != code_bytes or len(raw_scales) != metadata_bytes:
                    raise ValueError("truncated packed embedding row")
                codes = self.endpoint._unpack_signed_codes(
                    raw_codes, count=self.embedding_dim, spec=self.spec
                ).reshape(groups, int(self.payload.group_size))
                scales = torch.frombuffer(
                    bytearray(raw_scales), dtype=torch.float16
                ).reshape(groups)
                decoded.append(
                    (codes.float() * scales.float().unsqueeze(-1)).reshape(-1)
                )
        return torch.stack(decoded)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty(
                (*input_ids.shape, self.embedding_dim),
                device=input_ids.device,
                dtype=self.output_dtype,
            )
        minimum = int(input_ids.min().item())
        maximum = int(input_ids.max().item())
        if minimum < 0 or maximum >= self.num_embeddings:
            raise IndexError("embedding index out of range")
        unique, inverse = torch.unique(input_ids.detach().cpu(), sorted=True, return_inverse=True)
        rows = self._rows([int(item) for item in unique.tolist()])
        result = rows.index_select(0, inverse.reshape(-1)).reshape(
            *input_ids.shape, self.embedding_dim
        )
        return result.to(device=input_ids.device, dtype=self.output_dtype)


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    setattr(parent, child_name, replacement)


def _load_raw_tensor(endpoint: Any, path: Path) -> torch.Tensor:
    rows = [value for _, value in endpoint.iter_raw_tensor_rows(path, row_chunk=256)]
    return torch.cat(rows, dim=0)


@torch.no_grad()
def load_packed_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    row_chunk: int = 64,
    verify_hashes: bool = True,
    wal_linear_class: type[nn.Module] = PackedWALLinear,
    endpoint_linear_class: type[nn.Module] = PackedEndpointLinear,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load Qwen3 with packed operators and no dense Transformer weights."""

    started = time.monotonic()
    root, manifest, base_root, base_manifest = load_manifests(checkpoint)
    endpoint, codec = bundled_runtime(root)
    if verify_hashes:
        for row in manifest["checkpoint_files"]:
            member = root / str(row["file"])
            if member.stat().st_size != int(row["bytes"]) or _sha256(member) != row["sha256"]:
                raise ValueError(f"checkpoint integrity failure: {row['file']}")
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    requested = torch.device(device)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    base_by_name = {str(row["name"]): row for row in base_manifest["matrices"]}
    overlay_by_name = {str(row["name"]): row for row in manifest["overlays"]}
    original_modules = dict(model.named_modules())
    for weight_name, row in base_by_name.items():
        module_name = weight_name.removesuffix(".weight")
        original = original_modules[module_name]
        replacement = wal_linear_class(
            base_root / str(row["file"]),
            endpoint=endpoint,
            codec=codec,
            overlay_path=(
                None
                if weight_name not in overlay_by_name
                else root / str(overlay_by_name[weight_name]["file"])
            ),
            in_features=int(original.in_features),
            out_features=int(original.out_features),
            row_chunk=row_chunk,
            output_dtype=dtype,
        )
        _replace_module(model, module_name, replacement)

    endpoints = {str(row["name"]): row for row in base_manifest["endpoints"]}
    embedding_row = endpoints["model.embed_tokens.weight"]
    head_row = endpoints["lm_head.weight"]
    model.model.embed_tokens = PackedEmbedding(
        base_root / str(embedding_row["file"]), endpoint=endpoint, output_dtype=dtype
    )
    model.lm_head = endpoint_linear_class(
        base_root / str(head_row["file"]),
        endpoint=endpoint,
        row_chunk=row_chunk,
        output_dtype=dtype,
    )

    buffer_snapshots = {
        name: value.detach().contiguous().cpu().clone()
        for name, value in model.named_buffers()
        if not value.is_meta
    }
    model.to_empty(device=requested)
    model.to(dtype=dtype)
    buffers = dict(model.named_buffers())
    for name, snapshot in buffer_snapshots.items():
        buffers[name].copy_(snapshot.to(device=requested, dtype=buffers[name].dtype))
    if hasattr(model.model.rotary_emb, "inv_freq"):
        model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.inv_freq

    parameters = dict(model.named_parameters())
    for row in base_manifest["unquantized_tensors"]:
        name = str(row["name"])
        if name not in parameters:
            raise ValueError(f"raw parameter absent after packed replacement: {name}")
        value = _load_raw_tensor(endpoint, base_root / str(row["file"]))
        parameters[name].copy_(value.to(device=requested, dtype=dtype))

    model.eval()
    dense_body_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".layers." in name and name.endswith(".weight") and parameter.ndim == 2
    )
    report = {
        "status": "ok",
        "checkpoint": str(root),
        "device": str(requested),
        "dtype": str(dtype),
        "body_packed_matrices": len(base_by_name),
        "walb2_matrices": len(overlay_by_name),
        "dense_body_matrix_parameters": int(dense_body_parameters),
        "full_bf16_body_materialized": False,
        "temporary_decode_scope": f"at most {row_chunk} output rows",
        "load_seconds": time.monotonic() - started,
    }
    return model, report


def operator_check(
    checkpoint: str | Path,
    *,
    device: str,
    row_chunk: int,
    matrix_name: str | None,
    seed: int,
) -> dict[str, Any]:
    root, manifest, base_root, base_manifest = load_manifests(checkpoint)
    endpoint, codec = bundled_runtime(root)
    base_by_name = {str(row["name"]): row for row in base_manifest["matrices"]}
    overlay_by_name = {str(row["name"]): row for row in manifest["overlays"]}
    if matrix_name is None:
        matrix_name = next(name for name in base_by_name if name in overlay_by_name)
    row = base_by_name[matrix_name]
    shape = tuple(int(item) for item in row["shape"])
    module = PackedWALLinear(
        base_root / str(row["file"]),
        endpoint=endpoint,
        codec=codec,
        overlay_path=(
            None
            if matrix_name not in overlay_by_name
            else root / str(overlay_by_name[matrix_name]["file"])
        ),
        in_features=shape[1],
        out_features=shape[0],
        row_chunk=row_chunk,
        output_dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((2, shape[1]), generator=generator, dtype=torch.float32).to(device)
    started = time.monotonic()
    actual = module(value).float()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    packed_seconds = time.monotonic() - started

    dense_parts = []
    for _, rows in endpoint.iter_dequantized_matrix_rows(
        base_root / str(row["file"]), row_chunk=row_chunk, dtype=torch.float32
    ):
        dense_parts.append(rows)
    dense = torch.cat(dense_parts)
    if matrix_name in overlay_by_name:
        dense.add_(
            codec.decode_binary_lowrank_bundle(
                codec.read_binary_lowrank_bundle(
                    root / str(overlay_by_name[matrix_name]["file"])
                ),
                dtype=torch.float32,
            )
        )
    expected = value.float() @ dense.to(device).transpose(0, 1)
    error = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-6)
    return {
        "status": "ok",
        "matrix": matrix_name,
        "shape": list(shape),
        "device": device,
        "row_chunk": row_chunk,
        "packed_seconds": packed_seconds,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "mean_relative_error": float((error / denominator).mean().item()),
        "dense_body_materialized_by_runtime": False,
        "dense_matrix_materialized_only_by_test_oracle": True,
    }


def smoke(
    checkpoint: str | Path,
    *,
    device: str,
    row_chunk: int,
    prompt: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model, report = load_packed_model(
        checkpoint, device=device, dtype=dtype, row_chunk=row_chunk, verify_hashes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.monotonic()
    logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    report["forward"] = {
        "seconds": time.monotonic() - started,
        "input_tokens": int(encoded["input_ids"].numel()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "argmax_token_id": int(logits.argmax(dim=-1).item()),
        "argmax_token": tokenizer.decode(logits.argmax(dim=-1).tolist()),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if str(device).startswith("cuda")
            else None
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("operator-check")
    check.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CHECKPOINT))
    check.add_argument("--device", default="cpu")
    check.add_argument("--row-chunk", type=int, default=64)
    check.add_argument("--matrix")
    check.add_argument("--seed", type=int, default=20260806)
    run = sub.add_parser("smoke")
    run.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CHECKPOINT))
    run.add_argument("--device", default="cpu")
    run.add_argument("--row-chunk", type=int, default=64)
    run.add_argument("--prompt", default="Hello")
    args = parser.parse_args()
    if args.command == "operator-check":
        result = operator_check(
            args.checkpoint,
            device=args.device,
            row_chunk=args.row_chunk,
            matrix_name=args.matrix,
            seed=args.seed,
        )
    else:
        result = smoke(
            args.checkpoint,
            device=args.device,
            row_chunk=args.row_chunk,
            prompt=args.prompt,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
