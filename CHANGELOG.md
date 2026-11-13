# Changelog

Notable changes in each Matricxon release. Security fixes are marked **Security**. Full release notes and news:
https://bytepenny.com/rss/

## Unreleased

- multilingual-e5-small (cstr's GGUF) embeds again: its Unigram vocabulary is labeled `llama`, so it got the BPE tokenizer, which failed on any newline (`KeyError: 10`) and split words differently from the model. A `llama` vocabulary without byte tokens now uses the Unigram tokenizer (matches the HF tokenizer on all 15 test prompts), and the BPE one falls back to `<unk>` instead of failing.
- Block reuse: `POST /api/chat` takes `"cache_reuse": {"mode": "blocks", "min_tokens": 32}`. A prompt changed in the middle (a note edited, a skill added or removed) then keeps the cached parts after the change instead of re-reading them: each run of at least `min_tokens` equal tokens found in any cached chat is copied to its new position with its keys re-rotated, and only the tokens between runs are read. Approximate, since a copied part was read next to other text; the prefix and every token read stay exact. The default (`"exact"`) reuses only a shared prefix, as before. For plain KV caches of models whose keys can be re-rotated (Llama, Mistral 3, Qwen 2, Gemma 4); other models and prompts with images that can't be pooled read as usual.

- Saved prompt caches grow by pieces: after a reply only the new tokens are written to disk, not the whole chat again (a 2,500-token Gemma 4 chat rewrote about 225 MB per reply; a short reply's piece is a few MB). A chat is a chain of one full copy and up to 8 pieces; a new full copy is written when the pieces reach half the base, the chat's start changed (an edit, a rewind, a trimmed history), the chain isn't whole on disk, or the model keeps a recurrent state. Restores need the whole chain, checked piece by piece; limits remove chains whole; a chain missing a piece is cleaned after 10 minutes. Files saved before still restore.

- A pinned model on the CPU is copied into RAM when it loads, instead of being mapped from its file, so its weights are never read from disk again. Mapped pages are dropped when RAM is tight, and reading them back from a slow disk (a USB stick behind NFS) made a short reply wait about a minute. A model loaded before it was pinned is loaded again into RAM once its replies are done.

- A reply whose client disconnects stops its work: it used to keep reading a long prompt for minutes (a deleted chat's reply kept both CPU cores busy). Every reply now gets an id to stop it by.

- While a long prompt is still being read, `POST /api/chat` streams a `prompt_progress` line (tokens from the cache, tokens read so far, the layer being read) each time the read moves on, so a client can tell a slow read from a stuck one. The final line adds `prompt_cached_count`: of `prompt_eval_count`, the tokens reused from the prompt cache.

- Gemma 4 reads images: put its image file (`mmproj-*.gguf`, e.g. `unsloth/gemma-4-E2B-it-GGUF:mmproj-F16`) next to the model and it reports `vision`. The `gemma4v` image encoder matches Hugging Face's to 4 decimals on three image shapes (tools/oracle/validate_gemma4_vision.py). Images use 70 tokens each.
- A chat with an image is reused from the prompt cache for a model whose image placeholders are unique per picture (Gemma 4): the next message, even "OK", reads only what is new instead of the whole chat (163 of 174 tokens reused, 4 s instead of over a minute on a CPU). Encoded images are kept, so a picture sent again isn't re-encoded.

- Multilingual embedding models: the SentencePiece Unigram tokenizer (`tokenizer.ggml.model = "t5"`) of XLM-RoBERTa encoders such as `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5` and `bge-m3`. Matches the Hugging Face tokenizer on Hebrew, Arabic, Russian, Japanese and English test sentences, and the model's vectors match the original at 0.99+ (Q4_K_M).
- `GET /api/tags` and `/api/show` mark a model whose tokenizer Matricxon can't build with `tokenizer_unsupported`, instead of listing it as usable and failing on its first request.

## 2.1.2 (2026-11-11)

- Wider CPU instructions in the native kernels (AVX2, VNNI, ARM dot product) are on by default for a fresh install (`MATRICXON_WIDE_SIMD`); a CPU without them uses the plain kernels, and an existing `.env` value is kept.
- A shared folder that is missing, not mounted, not readable or behind a stale NFS handle now stops the start with one plain line naming the folder and the reason, on every node, instead of a Python error trace. The cluster readiness check no longer fails on such a folder.
- Time-series forecasting: `POST /api/forecast` runs Google TimesFM 2.5 (`timesfm25` GGUF). It returns a point forecast, the mean and nine quantiles per step, up to 128 steps ahead, and matches Hugging Face `transformers`. A forecasting model lists only the `forecast` capability and can't be used for chat.

## 2.1.1 (2026-11-08)

- Pinned models (`MATRICXON_PINNED_MODELS`): loaded in the background after start and never unloaded for being idle or to make room. A pin that can't load is tried 3 times, a minute apart, then its memory is used as usual. Each pin can choose RAM, GPU VRAM or auto (`tag|ram`, `tag|vram`), also on a server that otherwise runs on the CPU or the GPU. Set live with `PUT /api/runtime-settings` (`pinned_models`, saved in `.env`); `/api/node/load` and `/api/ps` report pins and their state. Deleting a model removes its pin; a pin whose model is gone fails at once instead of retrying.
- `/api/node/load` also reports total RAM, the GPU's memory and the model limit.
- `scripts/start.sh` refuses to run as root with a clear message naming the user to start it as (`MATRICXON_ALLOW_ROOT=1` for containers).
- The repository now ships its `.gitignore`: a clone no longer lists `.env`, caches and data as new files.
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
