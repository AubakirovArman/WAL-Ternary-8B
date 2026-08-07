# Security and integrity

## Trust boundary

The checkpoint is treated as immutable data. The public CLI downloads it with
Hugging Face tooling and uses `trust_remote_code=False` for the tokenizer.
Runtime implementation code comes from this signed/reviewable source tree.

## Fail-closed cache policy

The derived hardware cache is accepted only after matching:

1. canonical parent checkpoint manifest;
2. cache manifest and matrix count;
3. ABI/layout identity and byte order;
4. converter identity;
5. source lineage for every matrix;
6. cache file length and SHA-256.

Verification failure is not recoverable by silently falling back to stale
bytes. Rebuild the cache from the intended model revision.

## Operational guidance

- Do not use `--skip-hashes` outside controlled diagnostics.
- Keep model and cache directories read-only for serving users.
- Do not execute scripts from untrusted model repositories.
- Pin model revision and runtime commit in production experiments.
- Treat generated text as untrusted input before passing it to shells, tools or
  external systems.
- Review Qwen3 limitations, applicable laws and downstream safety requirements.

## Reporting

Open a GitHub issue with platform, exact command, runtime commit, model
revision, logs with secrets removed, and whether the problem reproduces with
hash verification enabled.
