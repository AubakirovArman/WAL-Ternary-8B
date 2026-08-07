# WAL-Ternary-8B physical format

## Transformer body

For every group of 128 weights, the base operator is:

```text
W_base = alpha * T + beta * R
T[i] ∈ {-1, 0, +1}
R has exactly 8 non-zero signed positions in each group
```

The hardware cache stores:

- four 2-bit ternary symbols per byte;
- eight uint8 residual positions per group;
- one byte of residual signs per group;
- FP16 alpha and beta per group.

The sparse lane is part of the base format; calling this pure T3 is incorrect.

## WALB2 correction

A corrected matrix adds one or more paths:

```text
DeltaW(x) = diag(row) · U_sign · diag(latent)
            · V_sign · diag(column) · x
```

`U_sign` and `V_sign` contain only {-1,+1} and are bit-packed. Row, latent
and column scales are positive FP16 values. V77 uses WALB2 corrections in 213
of 252 body matrices.

## Endpoints

- embedding: symmetric INT3 grouped by 128;
- LM head: symmetric INT4 grouped by 128;
- norms and small tensors: BF16.

## Accounting

- unique parameters: 8,190,735,360;
- serialized model bytes: 2,950,747,732;
- serialized model rate: 2.882034496 BPW;
- complete v0.2 tree bytes: 2,967,521,309;
- complete tree rate: 2.898417472 BPW.

BPW is total serialized bytes multiplied by eight and divided by the original
parameter count. It includes metadata where stated as complete-tree BPW.

## Runtime equation

```text
y = (T3 + sparse-k8) x + Σ U_r (V_r^T x)
```

The optimized runtime evaluates this equation from packed bytes. Temporary
activation buffers and KV cache are allowed; a persistent dense body matrix is
not created.

## Hardware cache

`.walhw` is a deterministic compute layout, not another model checkpoint.
Its attestation binds:

- canonical checkpoint manifest SHA-256;
- converter SHA-256;
- compute ABI and endianness;
- per-matrix source and cache SHA-256;
- matrix manifest root and byte counts.

A cache from a different checkpoint or ABI must be rejected.
