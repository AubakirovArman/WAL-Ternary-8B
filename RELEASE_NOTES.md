# WAL-Ternary-8B v0.2.0

Released: 2026-08-07

## Model

- V77 passes all four original project gates.
- Complete checkpoint: 2.898417 BPW.
- C4 PPL 20.1743; MMLU-Redux 59.29%; GSM8K strict 76.57%.
- Immutable model weights remain hosted at
  `armanibadboy/WAL-Ternary-8B`.

## Runtime

- Stable `wal-runtime` CLI.
- Direct-packed CUDA V18 with CUDA Graph.
- Native CPU V4 with x86 AVX-512 and ARM64/NEON paths.
- Portable CPU/CUDA/MPS correctness fallback.
- Deterministic, fail-closed `.walhw` cache preparation.
- H200 short-context wheel smoke: 99.24 tok/s, 3.658 GB peak allocated.
- Isolated wheel CPU smoke: exact accepted first four tokens.

## Known limitations

- Apple Silicon needs physical validation.
- CPU prefill is not yet optimized.
- First execution JIT-compiles native extensions.
- IFEval packed arithmetic remains more sensitive than C4/MMLU/GSM8K.
