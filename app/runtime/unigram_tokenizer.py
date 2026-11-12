import unicodedata

from app.gguf.metadata import GGUFMetadata

_NORMAL_TOKEN_TYPE = 1
_USER_DEFINED_TOKEN_TYPE = 4
_SPACE_MARKER = "▁"
# Below every real piece's score (the lowest is about -20): a character with no piece becomes <unk>
# only when nothing else covers it, as SentencePiece's own unknown penalty does.
_UNKNOWN_SCORE = -100.0


class UnigramTokenizer:
    """SentencePiece Unigram tokenizer for GGUF's `tokenizer.ggml.model = "t5"` (llama.cpp's UGM
    vocab), used by XLM-RoBERTa-based encoders such as `paraphrase-multilingual-MiniLM-L12-v2`,
    `multilingual-e5` and `bge-m3`.

    Unlike BPE there are no merges: each piece has a log-probability (`tokenizer.ggml.scores`) and
    a word is split into the pieces with the highest total score (Viterbi). The text is normalized
    first (NFKC: Python's table stands in for `precompiled_charsmap`), extra spaces are collapsed,
    a "▁" marks each word's start and one is added in front (`add_space_prefix`). Encodes as
    `<s> pieces </s>`, the single-sequence input of these encoders. Cross-checked against the HF
    tokenizer in tools/oracle/validate_unigram_tokenizer.py (15 of 15 in 8 scripts).
    """

    def __init__(self, metadata: GGUFMetadata) -> None:
        tokens: list[str] = metadata.require("tokenizer.ggml.tokens")
        scores: list[float] = metadata.require("tokenizer.ggml.scores")
        token_types: list[int] = metadata.require("tokenizer.ggml.token_type")
        self._id_to_token = tokens
        # Control tokens (<s>, </s>, <pad>) and <unk> never come out of splitting text.
        self._pieces: dict[str, tuple[int, float]] = {
            token: (i, score)
            for i, (token, score, token_type) in enumerate(
                zip(tokens, scores, token_types, strict=True)
            )
            if token_type in (_NORMAL_TOKEN_TYPE, _USER_DEFINED_TOKEN_TYPE)
        }
        self._longest_piece = max(len(piece) for piece in self._pieces)
        self.cls_token_id: int = metadata.require("tokenizer.ggml.bos_token_id")
        self.sep_token_id: int = metadata.get(
            "tokenizer.ggml.seperator_token_id"
        ) or metadata.require("tokenizer.ggml.eos_token_id")
        self.unk_token_id: int = metadata.require("tokenizer.ggml.unknown_token_id")
        self._add_space_prefix = bool(metadata.get("tokenizer.ggml.add_space_prefix", True))
        self._collapse_spaces = bool(metadata.get("tokenizer.ggml.remove_extra_whitespaces", True))

    def encode(self, text: str) -> list[int]:
        token_ids = [self.cls_token_id]
        for word in self._words(text):
            token_ids.extend(self._best_split(word))
        token_ids.append(self.sep_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Debug/inspection convenience: only encode is on the request path."""
        text = "".join(
            self._id_to_token[i]
            for i in token_ids
            if i not in (self.cls_token_id, self.sep_token_id)
        )
        return text.replace(_SPACE_MARKER, " ").strip()

    def _words(self, text: str) -> list[str]:
        """Normalized words, each starting with "▁" (the first only with add_space_prefix)."""
        text = unicodedata.normalize("NFKC", text)
        text = "".join(
            " " if char.isspace() else char
            for char in text
            if unicodedata.category(char) != "Cc" or char.isspace()
        )
        words = text.split() if self._collapse_spaces else text.split(" ")
        marked = [_SPACE_MARKER + word for word in words if word]
        if marked and not self._add_space_prefix:
            marked[0] = marked[0][len(_SPACE_MARKER) :]
        return marked

    def _best_split(self, word: str) -> list[int]:
        """Viterbi: the split of `word` into vocab pieces with the highest total score. A character
        no piece covers is <unk> (consecutive ones merge into one, as SentencePiece does)."""
        best: list[tuple[float, int, int]] = [(0.0, 0, -1)] + [(float("-inf"), 0, -1)] * len(word)
        for start in range(len(word)):
            score_here = best[start][0]
            if score_here == float("-inf"):
                continue
            matched = False
            for end in range(start + 1, min(len(word), start + self._longest_piece) + 1):
                piece = self._pieces.get(word[start:end])
                if piece is None:
                    continue
                matched = matched or end == start + 1
                candidate = score_here + piece[1]
                if candidate > best[end][0]:
                    best[end] = (candidate, start, piece[0])
            if not matched and score_here + _UNKNOWN_SCORE > best[start + 1][0]:
                best[start + 1] = (score_here + _UNKNOWN_SCORE, start, self.unk_token_id)
        token_ids: list[int] = []
        end = len(word)
        while end > 0:
            _score, start, token_id = best[end]
            if not (
                token_id == self.unk_token_id and token_ids and token_ids[-1] == self.unk_token_id
            ):
                token_ids.append(token_id)
            end = start
        token_ids.reverse()
        return token_ids
