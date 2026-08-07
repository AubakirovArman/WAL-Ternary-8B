# WAL-Ternary-8B: converting Qwen3-8B into an executable 2.898-BPW low-bit system

## Abstract

WAL-Ternary-8B V77 is a self-contained low-bit conversion of Qwen3-8B. All 252
Transformer-body linear matrices are stored as a T3+sparse-k8 base; 213 also
carry packed binary low-rank WALB2 corrections. The embedding is INT3, the LM
head INT4, and the complete checkpoint occupies 2.898417 bits per original
parameter. No Neutrino tensor or dense Qwen checkpoint is required at inference.

Pure post-training ternarization collapsed to C4 PPL ≈7952.54. Sparse-k8
recovered the model to roughly PPL 41, and the complete V73 residual system
reached 28.58. A sequence of causal experiments showed that better local
operator fidelity does not imply better autoregressive trajectories. The R12
breakthrough jointly calibrated 465 already serialized amplitudes without
adding bytes. V77 reached C4 PPL 20.1743, MMLU-Redux 59.29%, and GSM8K strict
76.57%, passing all four original project gates.

A direct-packed runtime then removed the need to materialize a dense body.
Steady-state H200 decode improved from 4.62 to 72.41 tok/s at 3.93 GB peak
VRAM; a short-context 256-KV run reached 99.36 tok/s at 3.66 GB. Native Xeon
CPU decode reached 5.93 tok/s at 4.12 GB RSS.

## 1. Problem

The goal was not merely to produce a smaller archive. It was to preserve a
trained Qwen3-8B's knowledge and reasoning in a roughly 3-BPW representation
that could execute directly on CPU and GPU.

This differs from training a ternary model from the beginning. BitNet b1.58
integrates {-1,0,+1} weights into training. WAL starts from an already trained
BF16 model and must preserve its function under a severe post-training
representation constraint.

The original preregistered success criteria were:

| Gate | Requirement |
|---|---:|
| Complete checkpoint | ≤2.952031 BPW |
| C4 PPL | ≤22.5637 |
| MMLU-Redux | ≥57.28% |
| GSM8K strict | ≥58.66% |

## 2. Representation

For every group of 128 body weights:

```text
W_base = alpha * T + beta * R
T[i] ∈ {-1, 0, +1}
R contains exactly eight signed non-zero sites
```

For 213 matrices:

```text
DeltaW(x) = Σ diag(row) U_sign diag(latent)
              V_sign diag(column) x
```

`U_sign` and `V_sign` are binary and bit-packed; scale vectors are FP16.
The remaining 39 matrices use only the base. Embedding is INT3/g128 and the LM
head INT4/g128.

Serialized model bytes are 2,950,747,732 for 8,190,735,360 original parameters,
or 2.882034 BPW. The complete release tree is 2.898417 BPW.

WAL is therefore a ternary-derived hybrid, not a pure 1.58-bit model.

## 3. Experimental discipline

Each material experiment used a frozen design card defining:

- hypothesis and causal mechanism;
- allowed parameters and byte budget;
- train/selection/confirmation firewall;
- metrics, gates and stopping conditions;
- immutable checkpoint and dataset identities.

Generation was evaluated with paired wrong-to-correct and correct-to-wrong
transitions. Local weight or activation reconstruction was never sufficient
for promotion without a full-model test.

## 4. From pure T3 to V73

Representation ablation established:

| Representation | C4 PPL |
|---|---:|
| Pure T3 | ≈7952.54 |
| T3+sparse-k8 | ≈41 |
| T3+sparse-k8+WALB2 V73 | 28.5811 |

The system was strongly non-additive. Sparse-k8 delivered the first large
recovery; WALB2 depended on the co-adapted base and could not independently
rescue pure T3.

V73 achieved full body coverage at 2.898417 BPW, MMLU-Redux 60.36%, GSM8K
strict 75.28%, and GSM8K flexible 76.57%. Only C4 remained above the original
gate.

## 5. Prompt protocol and trajectory diagnosis

An early bare-prompt GSM8K run gave 0% strict. Correct Qwen3 non-thinking chat
format recovered about 73% on the diagnostic sample, proving that much of the
apparent collapse was a protocol error.

The residual gap remained real. V73 showed about 85.9% top-1 agreement with
BF16, 0.42–0.50 nat/token KL, slightly lower entropy, early semantic divergence
and excessive confidence after choosing a different branch.

## 6. Local repair failures

R7d recalibrated existing L4 `gate_proj` row scales. Across three disjoint
sets it reduced local failure KL by about 5%, cut gate pre-activation error by
about 35%, retained 53–55% of a BF16 gate oracle's effect, and added zero bytes.
Yet full generation produced 14 repairs and 18 regressions among 600 answers.

R8 showed that global shrinkage, rank-64 constrained directions and the full
36,864-dimensional row-scale space could not guarantee sequence-safe repair.
R9 rebuilt continuous same-rank directions; R10 joined L4 gate and L10 V/O
causal nodes. Local fitting repeatedly passed while fresh full-generation
transfer failed.

The accumulated conclusion was:

> Operator recovery and autoregressive trajectory recovery are different
> optimization problems.

## 7. R12 global co-adaptation

R12 returned to the only failed top-level gate: C4. It jointly optimized 465
existing foldable amplitudes—252 base coefficients and 213 WALB2
coefficients—on sealed train-only C4 windows. No codes, sparse positions,
factors, ranks or metadata were added.

V76 R12-A delivered the large C4 recovery but exposed a state-dependent
termination issue. One authorized R12-B1-Lite fit recalibrated the same
amplitudes with stop/continue and preservation anchors. Folding those scales
produced V77.

## 8. V77 results

| Metric | V73 | V77 |
|---|---:|---:|
| Complete BPW | 2.898417 | **2.898417** |
| C4 PPL | 28.5811 | **20.1743** |
| MMLU-Redux | **60.36%** | 59.29% |
| GSM8K strict | 75.28% | **76.57%** |
| GSM8K flexible | 76.57% | **77.10%** |
| IFEval strict, cap 256 | 48.24% | **51.57%** |
| IFEval repeated tail | 35.86% | **28.65%** |

V77 passes the four original gates. Later V73-relative diagnostics recorded a
1.07-point MMLU trade-off and a 2.40-point increase in bounded max-length hits.
Those findings remain public; they do not retroactively redefine the original
absolute objective.

## 9. Direct-packed inference

The early reference loader reconstructed BF16 weights. The new runtime directly
computes:

```text
y = (T3+sparse-k8)x + Σ U_r(V_r^T x)
```

A deterministic approximately 2.3 GB `.walhw` cache rearranges packed symbols,
sparse positions/signs and FP16 scales for computation. It remains low-bit and
is cryptographically bound to the canonical checkpoint and runtime ABI.

GPU progress:

| Stage | H200 decode |
|---|---:|
| first direct-packed | 4.62 tok/s |
| first WALB2 fusion | 5.13 tok/s |
| V18 warm non-graph | about 32 tok/s |
| V18 CUDA Graph, KV 2048 | **72.41 tok/s** |
| V18 auto KV 256 | **99.36 tok/s** |

Peak GPU memory is 3.66–3.93 GB instead of the former 12.60 GB path. Native
CPU V4 reaches 5.93 tok/s on a 64-thread Xeon at 4.12 GB RSS.

## 10. Runtime parity

| Metric | Materialized V77 | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | 20.2156 |
| MMLU-Redux | 59.29% | 59.26% |
| GSM8K strict | 76.57% | 76.19% |
| GSM8K flexible | 77.10% | 76.35% |

IFEval is more numerically sensitive. A confirmed V18 deployment profile with a
512-token cap and live repeated-tail guard improved over the frozen 256-token
materialized profile on untouched documents, but still trails the materialized
512-token reference. Exact logit parity is not claimed.

## 11. Lessons

1. Pure post-training ternarization did not preserve an 8B model's function.
2. Sparse and low-rank lanes must be co-adapted rather than optimized alone.
3. Better weight/operator/KL metrics do not ensure better full sequences.
4. Baseline-correct trajectories must be explicitly protected.
5. Large quality recovery can exist in already serialized scale capacity.
6. A compact file only becomes a compact model when kernels consume packed
   bytes directly.

## 12. Limitations

Apple ARM64/NEON is implemented but not physically validated. CPU prefill is
slow. CUDA performance is mainly validated on H200/SM90. IFEval packed
arithmetic remains sensitive. Broad coding, tool-use, safety, multilingual,
long-context and same-harness Neutrino comparisons remain incomplete.

## Conclusion

WAL-Ternary-8B demonstrates that a pretrained Qwen3-8B can be independently
converted into a self-contained 2.898417-BPW model with complete low-bit body
coverage, four-of-four original quality/rate gates, and direct-packed CPU/CUDA
execution without a persistent dense Transformer body.

The contribution is not a claim of pure 1.58-bit inference. It is a documented
construction showing which ingredients were necessary, which intuitive repair
methods failed, why global co-adaptation worked, and how physical compression
was turned into practical runtime memory and speed.

## References

1. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
2. Qwen. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B).
3. Ma et al. [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764), 2024.
4. Jin et al. [PARQ](https://arxiv.org/abs/2503.15748), 2025.
5. Fermion Research. [Neutrino-8B model card](https://huggingface.co/FermionResearch/Neutrino-8B), accessed 2026-08-07.
