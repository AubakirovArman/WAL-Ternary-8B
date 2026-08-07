# Benchmark and runtime evidence

Date of frozen V77 release battery: 2026-08-06 UTC. Runtime measurements:
2026-08-07 UTC.

## Model gate battery

| Metric | Protocol | V73 | V77 | Original gate |
|---|---|---:|---:|---:|
| Complete tree BPW | all serialized files | 2.898417 | **2.898417** | ≤2.952031 |
| C4 PPL | fixed 128×2048 slice, 262,016 scored tokens | 28.58 | **20.1743** | ≤22.5637 |
| MMLU-Redux | 30 subjects, 2,722 questions, 5-shot letter loglik | **60.36%** | 59.29% | ≥57.28% |
| GSM8K strict | 1,319 test items, Qwen chat, thinking off, greedy, cap 512 | 75.28% | **76.57%** | ≥58.66% |
| GSM8K flexible | same run | 76.57% | **77.10%** | diagnostic |
| IFEval strict | 541 prompts, thinking off, greedy, cap 256 | 48.24% | **51.57%** | diagnostic |
| IFEval loose | same run | 53.97% | **54.90%** | diagnostic |
| IFEval repeated tail | same run | 35.86% | **28.65%** | diagnostic |

V77 passes the four original absolute gates. The later V73-relative guardrails
remain in the audit record: MMLU was 1.07 percentage points lower and bounded
IFEval max-length rate was 2.40 points higher. They are not hidden and were not
used to rewrite the original scientific protocol.

## Direct-packed numerical parity

The direct runtime reads T3+sparse-k8, WALB2, INT3 and INT4 bytes without a
persistent dense body.

| Metric | Materialized V77 | Direct-packed V12 | Difference |
|---|---:|---:|---:|
| C4 PPL | 20.1743 | 20.2156 | +0.0413 |
| MMLU-Redux | 59.29% | 59.26% | -0.03 pp |
| GSM8K strict | 76.57% | 76.19% | -0.38 pp |
| GSM8K flexible | 77.10% | 76.35% | -0.75 pp |

IFEval was more sensitive: the original direct-packed cap-256 path measured
49.35% strict versus 51.57% materialized. This is why the project does not
claim exact logit equivalence.

## Selected IFEval deployment profile

Discovery used documents 0–99. Confirmation used untouched documents 100–540.
The selected profile is V18 batch-one greedy, cap 512, with a live repeated
16-token tail guard searching the preceding 80 tokens.

| Untouched 441 documents | Strict | Loose |
|---|---:|---:|
| Frozen materialized cap 256 | 52.38% | 55.56% |
| V18 cap 512 + live guard | **56.69%** | **60.54%** |
| Materialized cap 512 | 58.96% | 63.72% |

Against the frozen cap-256 reference, V18 produced 37 strict repairs versus 18
regressions and 40 loose repairs versus 18 regressions. It still trails the
materialized cap-512 reference by 2.27 strict and 3.17 loose points.

This is a deployment profile, not a new checkpoint. The frozen benchmark is
preserved.

## Runtime performance

| Path | Hardware/settings | Decode | Peak memory |
|---|---|---:|---:|
| First direct-packed prototype | H200 | 4.62 tok/s | 3.75 GB |
| First WALB2 fusion | H200 | 5.13 tok/s | about 3.75 GB |
| V18 CUDA Graph | H200, KV 2048 | **72.41 tok/s** | **3.93 GB** |
| V18 CLI short-context | H200, auto KV 256 | **99.36 tok/s** | **3.66 GB** |
| Native CPU V4 | Xeon, 64 threads | **5.93 tok/s** | **4.12 GB RSS** |

The H200 99.36 tok/s result measures 128 pure Graph replay decode steps after
compile, prefill and capture. The 72.41 tok/s result is the frozen long-cache
comparison. Neither result includes model download or first-run compilation.

## Reproducibility rules

- Record exact model revision and manifest SHA.
- Record backend, batch size, prompt/chat template, thinking mode and output cap.
- Record hardware, thread count, PyTorch/CUDA/compiler versions and KV length.
- Separate cold start, prefill and steady-state decode.
- Compare per-document/per-question outputs when diagnosing parity.
- Do not compare official numbers produced by another harness as if they were
  same-harness results.
