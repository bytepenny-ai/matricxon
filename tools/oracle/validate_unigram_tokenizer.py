"""Cross-checks UnigramTokenizer against the HF tokenizer of an XLM-RoBERTa encoder (twin of
validate_wordpiece_tokenizer.py). No memory cap: the GGUF header and a tokenizer.json fetch."""

import sys
from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.models.tokenizer_dispatch import build_tokenizer

# (GGUF, HF repo), built through build_tokenizer: e5's "llama"-labeled Unigram vocab is covered too.
_MODELS = [
    (
        "hf.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/"
        "paraphrase-multilingual-MiniLM-L12-118M-v2-Q4_K_M.gguf",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    (
        "hf.co/cstr/multilingual-e5-small-GGUF/multilingual-e5-small-iq4_xs.gguf",
        "intfloat/multilingual-e5-small",
    ),
]
_ROOTS = [Path("/mnt/mxshared/matricxon"), Path("/media/home/models/matricxon")]
_PROMPTS = [
    "Hello, world!",
    "what kind of a bird is that?",
    "איזו ציפור זו?",
    "מה גובה הגלים בתמונה?",
    "The quick brown fox jumps over the lazy dog.",
    "   leading and   extra   spaces   ",
    "café naïve résumé",
    "Ｆｕｌｌ－ｗｉｄｔｈ ＡＢＣ １２３",
    "Numbers: 12345 and 6.789, punctuation!! -- test.",
    "Привет, как дела?",
    "東京は日本の首都です。",
    "emoji test \U0001f389 and unicode ℝ ﬁ",
    "tab\tand\nnewline",
    "unbelievable antidisestablishmentarianism",
    "مرحبا بالعالم",
]


def main() -> int:
    failed = 0
    for relative, repo in _MODELS:
        gguf = next((root / relative for root in _ROOTS if (root / relative).exists()), None)
        if gguf is None:
            print(f"[SKIP] model not present on this machine: {relative}")
            continue
        print(f"== {repo}")
        ours = build_tokenizer(GGUFReader(gguf).read().metadata)
        failed += _check(ours, AutoTokenizer.from_pretrained(repo))
    return 1 if failed else 0


def _check(ours, hf) -> int:
    failed = 0
    for prompt in _PROMPTS:
        our_ids, hf_ids = ours.encode(prompt), hf(prompt)["input_ids"]
        ok = our_ids == hf_ids
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  ours: {our_ids}\n  hf:   {hf_ids}")
            print(f"  ours: {hf.convert_ids_to_tokens(our_ids)}")
            print(f"  hf:   {hf.convert_ids_to_tokens(hf_ids)}")
    print(f"{len(_PROMPTS) - failed}/{len(_PROMPTS)} match")
    return failed


if __name__ == "__main__":
    sys.exit(main())
