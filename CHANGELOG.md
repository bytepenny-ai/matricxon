# Changelog

Notable changes in each Matricxon release. Security fixes are marked **Security**. Full release notes and news:
https://bytepenny.com/rss/

## Unreleased

- **Security:** vulnerability reporting policy (SECURITY.md), CodeQL code scanning and Dependabot updates.
- Updated dependencies, including FastAPI 0.162 and Pydantic 2.14.
- `/api/version` reports the git commit next to the version.

## 2.1 (2026-11-03)

- OpenAI-compatible API, JSON and JSON Schema replies, min_p, penalties, stop texts and Prometheus metrics.
- **Security:** API keys protect the model routes.
- The AVX2 and ARM kernels got the fast prompt path. Large chat caches load in the background while starting.
- Improved version updates. Clearer start script errors, including shared storage problems.
- Native C kernel, security and cluster mode fixes.

## 2.0 (2026-10-11)

- Cluster mode: several nodes share models and chat caches.
- **Security:** saved chat caches on disk are encrypted with AES-256-GCM.
- GPU mode (experimental) on NVIDIA GPUs. MCP updates.
- Qwen 3.5, Gemma 4, Falcon, StarCoder 2 and Mixtral, with a faster shared mixture-of-experts layer.
- The full I-quant family: 25 quantization types. Defaults for a fresh install.

## 1.2 (2026-09-23)

- All dequantization moved to native C kernels, many times faster, with macOS support.
- New dequantization architectures. Qwen 2 and Qwen 3.

## 1.1 (2026-09-22)

- First public release, under AGPL-3.0.
