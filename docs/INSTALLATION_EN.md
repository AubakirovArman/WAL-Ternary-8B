# Complete installation and usage guide

## 1. Requirements

Common:

- 64-bit Linux, macOS arm64, or Windows through WSL2;
- Python 3.10–3.12;
- PyTorch 2.2 or newer;
- about 3 GB for the model plus about 2.3 GB for the derived compute cache;
- internet access for the first Hugging Face download.

NVIDIA:

- NVIDIA driver compatible with the installed PyTorch;
- CUDA Toolkit with `nvcc`;
- Ninja and a C++ compiler;
- validated reference: H200/SM90.

x86-64 CPU:

- C++17 compiler and Ninja;
- AVX-512 is preferred; AVX2/scalar fallback exists;
- 16 GB system RAM recommended for build and execution headroom.

Apple Silicon:

- native arm64 Python and PyTorch;
- Xcode Command Line Tools: `xcode-select --install`;
- the native NEON path is experimental until real-Mac validation is published.

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install '.[runtime]'
wal-runtime doctor
```

From GitHub without cloning:

```bash
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
```

## 3. Inspect the model

```bash
wal-runtime inspect armanibadboy/WAL-Ternary-8B
```

This downloads the immutable model snapshot if it is not already cached.

## 4. Build the compute cache

```bash
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

The destination must not already exist. This protects against accidental reuse
of a partially created cache. Keep `manifest.json` and `attestation.json`
with the `.walhw` files.

For an offline machine, copy both the Hugging Face snapshot and the complete
cache directory. Pass the local snapshot path instead of the model id.

## 5. Generate

NVIDIA:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cuda:0 --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Give a concise explanation of photosynthesis.' \
  --max-new-tokens 256
```

CPU:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cpu --backend cpu-v4 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Give a concise explanation of photosynthesis.'
```

Apple Silicon uses the CPU command. Start with the default thread count and
compare against `--threads 4`, `8` and the number of performance cores.

Portable correctness fallback:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cpu --backend portable --prompt 'Hello' --max-new-tokens 16
```

The portable path is intentionally slow and does not use the hardware cache.

## 6. Context and memory

`--max-cache-len 0` automatically chooses the smallest power of two that fits
the prompt and requested output, with a minimum of 256. Larger static KV cache
lengths increase GPU memory and can reduce single-stream speed.

Example:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --max-cache-len 2048 --max-new-tokens 512 --prompt '...'
```

## 7. Benchmark correctly

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --max-cache-len 2048 --tokens 128
```

The first invocation compiles extensions and captures a CUDA Graph. Report
hardware, PyTorch/CUDA versions, prompt length, KV cache length and whether the
run was warm. Do not compare results with different settings.

## 8. Reproducibility checks

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_runtime_cli.py -q
python -m build --wheel
```

`wal-runtime prepare` and each optimized loader verify hashes by default.
`--skip-hashes` exists only for controlled diagnostics and should not be used
for normal inference.

## 9. Troubleshooting

`nvcc not found`: install a CUDA Toolkit matching your environment and ensure
`nvcc --version` works.

`ninja not found`: run `python -m pip install ninja`.

CUDA out of memory: leave `--max-cache-len 0`, shorten the prompt/output, and
check that another process is not using the GPU.

Cache rejected: do not bypass verification. Remove the specific cache
directory and rebuild it from the intended model revision.

Slow first run: JIT compilation is expected. The following warm run should
reuse the extension cache.

Slow CPU prefill: this is a known v0.2 limitation; decode is considerably more
optimized than prompt ingestion.

Apple build failure: confirm that Python, PyTorch and the shell are all arm64
and that `xcode-select -p` succeeds.

## 10. Security

The model is loaded with `trust_remote_code=False`. The runtime treats the
checkpoint as data, verifies manifests and hashes, and fails closed for stale
hardware caches. As with any model, review prompts, outputs and deployment
permissions before exposing it to untrusted users.
