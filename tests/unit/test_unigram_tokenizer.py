"""UnigramTokenizer (`tokenizer.ggml.model = "t5"`, XLM-RoBERTa encoders) on a tiny vocab; the
real one is checked against HF's tokenizer in tools/oracle/validate_unigram_tokenizer.py. Also
tokenizer_supported, which marks a model whose tokenizer matricxon can't build."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.gguf.metadata import GGUFMetadata
from app.models import tokenizer_dispatch
from app.models.tokenizer_dispatch import build_tokenizer, tokenizer_supported
from app.runtime.unigram_tokenizer import UnigramTokenizer

# id: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, then pieces with log-probabilities.
_PIECES = [
    ("▁", -2.0),
    ("▁un", -3.0),
    ("believ", -4.0),
    ("able", -3.0),
    ("▁unbeliev", -6.0),
    ("a", -5.0),
    ("b", -5.0),
    ("l", -5.0),
    ("e", -5.0),
    ("▁hi", -2.0),
    ("!", -1.0),
]


def _metadata(**overrides) -> GGUFMetadata:
    tokens = ["<s>", "<pad>", "</s>", "<unk>", *(p for p, _ in _PIECES)]
    values = {
        "tokenizer.ggml.model": "t5",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": [0.0, 0.0, 0.0, 0.0, *(s for _, s in _PIECES)],
        "tokenizer.ggml.token_type": [3, 3, 3, 2, *([1] * len(_PIECES))],
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.seperator_token_id": 2,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.add_space_prefix": True,
        "tokenizer.ggml.remove_extra_whitespaces": True,
        **overrides,
    }
    return GGUFMetadata(values)


def _ids(*pieces: str) -> list[int]:
    tokens = [p for p, _ in _PIECES]
    return [tokens.index(p) + 4 for p in pieces]


def test_the_highest_scoring_split_wins():
    # "▁unbeliev"+"able" scores -9, better than "▁un"+"believ"+"able" (-10).
    assert UnigramTokenizer(_metadata()).encode("unbelievable") == [
        0,
        *_ids("▁unbeliev", "able"),
        2,
    ]


def test_spaces_are_collapsed_and_each_word_starts_with_the_marker():
    tokenizer = UnigramTokenizer(_metadata())
    assert tokenizer.encode("  hi   hi! ") == [0, *_ids("▁hi", "▁hi", "!"), 2]


def test_characters_no_piece_covers_become_one_unknown():
    assert UnigramTokenizer(_metadata()).encode("hi☃☃") == [0, *_ids("▁hi"), 3, 2]


def test_control_tokens_never_come_out_of_text():
    assert 0 not in UnigramTokenizer(_metadata()).encode("<s>")[1:-1]


def test_dispatch_builds_it_for_t5():
    assert isinstance(build_tokenizer(_metadata()), UnigramTokenizer)


@pytest.mark.parametrize(
    ("model", "supported"), [("t5", True), ("bert", True), ("rwkv", False), (None, True)]
)
def test_tokenizer_supported(monkeypatch, tmp_path, model, supported):
    path = tmp_path / f"{model}.gguf"
    path.write_bytes(b"x")
    values = {} if model is None else {"tokenizer.ggml.model": model}

    class FakeReader:
        def __init__(self, _path: Path) -> None:
            pass

        def read(self):
            return SimpleNamespace(metadata=GGUFMetadata(values))

    monkeypatch.setattr(tokenizer_dispatch, "GGUFReader", FakeReader)
    assert tokenizer_supported(path) is supported


def test_an_unreadable_file_never_breaks_the_listing(tmp_path):
    assert tokenizer_supported(tmp_path / "missing.gguf") is True
