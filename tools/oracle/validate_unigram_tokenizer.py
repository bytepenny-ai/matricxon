"""Cross-checks UnigramTokenizer against the HF tokenizer of an XLM-RoBERTa encoder (twin of
validate_wordpiece_tokenizer.py). No memory cap: the GGUF header and a tokenizer.json fetch."""

import sys
from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.unigram_tokenizer import UnigramTokenizer

_GGUF = Path(
    "/mnt/mxshared/matricxon/hf.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/"
    "paraphrase-multilingual-MiniLM-L12-118M-v2-Q4_K_M.gguf"
)
_HF_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
    if not _GGUF.exists():
        print(f"[SKIP] model not present on this machine: {_GGUF}")
        return 0
    ours = UnigramTokenizer(GGUFReader(_GGUF).read().metadata)
    hf = AutoTokenizer.from_pretrained(_HF_REPO)
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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
