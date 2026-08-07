# WAL-Ternary-8B

[English](README.md) · [Русский](README_RU.md) · [Қазақша](README_KK.md)

WAL-Ternary-8B V77 is an independently constructed low-bit version of
`Qwen/Qwen3-8B`. The complete checkpoint occupies **2.898417 bits per
original parameter (BPW)**. All 252 Transformer-body matrices execute directly
from the packed representation; dense Qwen or Neutrino weights are not needed.

Weights and a copy of the runtime are hosted together at
[`armanibadboy/WAL-Ternary-8B`](https://huggingface.co/armanibadboy/WAL-Ternary-8B).

> Research release. CUDA and x86-64 CPU are validated. Apple ARM64/NEON code is
> included, but physical Apple Silicon validation is still pending.

## Representation

```text
W(x) = (alpha · T + beta · R)x + Σ Uᵣ(Vᵣᵀx)
T ∈ {-1, 0, +1}
R = exactly 8 signed residual sites per group of 128 weights
```

- 252/252 body matrices: T3 + sparse-k8 base.
- 213 matrices: packed binary low-rank WALB2 correction.
- 39 matrices: base only.
- Embedding: symmetric INT3/g128.
- LM head: symmetric INT4/g128.
- Norm and small tensors: BF16.
- Model payload: 2.882034 BPW; complete tree: 2.898417 BPW.

This is a ternary-derived hybrid, not a pure 1.58-bit BitNet checkpoint.

## Verified results

| Frozen original gate | Required | V77 |
|---|---:|---:|
| Complete checkpoint | ≤2.952031 BPW | **2.898417 BPW** |
| C4 perplexity | ≤22.5637 | **20.1743** |
| MMLU-Redux, 5-shot | ≥57.28% | **59.29%** |
| GSM8K strict | ≥58.66% | **76.57%** |

Additional reference results: GSM8K flexible **77.10%**, IFEval strict
**51.57%**, IFEval loose **54.90%**.

Direct-packed parity:

| Metric | Materialized reference | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | **20.2156** |
| MMLU-Redux | 59.29% | **59.26%** |
| GSM8K strict | 76.57% | **76.19%** |
| GSM8K flexible | 77.10% | **76.35%** |

## Runtime

The runtime retains no dense INT8/BF16 Transformer body.

| Backend | Result | Memory | Status |
|---|---:|---:|---|
| H200, CUDA V18 Graph, 2048 KV | 72.41 tok/s | 3.93 GB | validated |
| H200, auto 256 KV | 99.36 tok/s | 3.66 GB | validated short-context run |
| Xeon CPU V4, 64 threads | 5.93 tok/s | 4.12 GB RSS | validated |
| Apple ARM64/NEON | — | — | implemented; Mac validation pending |

These are hardware-specific measurements, not universal guarantees.

## Install

From GitHub:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
wal-runtime doctor
```

From the copy bundled with the model:

```bash
git clone https://huggingface.co/armanibadboy/WAL-Ternary-8B
cd WAL-Ternary-8B/runtime
python -m pip install '.[runtime]'
wal-runtime doctor
```

Build the approximately 2.3 GB executable cache:

```bash
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

The cache stores packed ternary symbols, sparse positions/signs and FP16
scales—not a reconstructed dense weight tree. Parent, ABI, converter,
per-matrix lineage and SHA-256 hashes are verified fail-closed.

Generate on NVIDIA:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Explain why the sky is blue in one sentence.' \
  --max-new-tokens 256
```

Generate on CPU:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cpu-v4 --device cpu \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Explain photosynthesis.'
```

Benchmark:

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 --tokens 128
```

The first CUDA run JIT-compiles the included kernels and needs CUDA/NVCC and
Ninja. CPU needs a C++17 compiler. Apple Silicon needs native arm64 Python,
PyTorch and Xcode Command Line Tools; use `cpu-v4`. The portable MPS fallback
is `--backend portable --device mps`, not a fused Metal backend.

Complete guides:

- [English](docs/INSTALLATION_EN.md)
- [Русский](docs/INSTALLATION_RU.md)
- [Қазақша](docs/INSTALLATION_KK.md)

## CLI

```text
wal-runtime doctor
wal-runtime inspect MODEL
wal-runtime prepare MODEL --output CACHE
wal-runtime generate MODEL --hardware-cache CACHE --prompt TEXT
wal-runtime benchmark MODEL --hardware-cache CACHE --tokens 128
```

## Repository

```text
src/wal_tat/runtime/       CLI, validation and packed backends
src/wal_tat/runtime/_v18/  CUDA V18, CPU V4 and reference modules
src/wal_tat/runtime/experiments/cuda/ CUDA C++ kernels
tests/                     focused runtime tests
docs/                      format, benchmarks, guides and article
```

## Limitations

- Research runtime, not yet a production serving engine.
- First-run JIT; precompiled wheels are not yet published.
- CPU prefill still needs batched packed GEMM.
- Apple Silicon and non-SM90 CUDA validation are incomplete.
- IFEval remains numerically sensitive; see [benchmarks](docs/BENCHMARKS.md).
- Coding, tool use, safety, multilingual and long-context testing is limited.

The full history is in the [Russian research article](docs/RESEARCH_ARTICLE_RU.md)
and [English technical paper](docs/RESEARCH_ARTICLE_EN.md).

## License

Apache-2.0, following the upstream
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) license. No Neutrino
tensor is included. This project is independent of Alibaba/Qwen and Fermion
Research.

Built by Arman Aubakirov with an AI-assisted experimental workflow.
