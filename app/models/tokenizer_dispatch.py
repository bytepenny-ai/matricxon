from pathlib import Path

from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.models.handle import Tokenizer
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer
from app.runtime.tokenizer import GGUFTokenizer
from app.runtime.unigram_tokenizer import UnigramTokenizer
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer
from app.server.errors import UnsupportedArchitectureError


def build_tokenizer(metadata: GGUFMetadata) -> Tokenizer:
    """Dispatches on `tokenizer.ggml.model`, the key llama.cpp uses to pick a tokenizer: `"gpt2"`
    for the byte-level BPE decoder models use, `"bert"` for the WordPiece format of both local
    encoder models (checked against their GGUF metadata), `"llama"` for the
    SentencePiece-BPE of plain `llama`-arch models (M10, checked on a TinyLlama pull), `"gemma4"`
    for Gemma4's rank-based BPE without byte-remapping (M10, checked on a `google/gemma-4-12b-it`
    pull). A `"llama"` vocab with no byte tokens is Unigram mislabeled by an older converter (an
    XLM-RoBERTa encoder such as cstr's multilingual-e5-small): SentencePiece-BPE always has its 256
    byte-fallback tokens, and its scores are the Unigram log-probabilities."""
    tokenizer_model = metadata.require("tokenizer.ggml.model")
    if tokenizer_model == "llama" and not _has_byte_tokens(metadata):
        return UnigramTokenizer(metadata)
    if tokenizer_model == "gpt2":
        return GGUFTokenizer(metadata)
    if tokenizer_model == "bert":
        return WordPieceTokenizer(metadata)
    if tokenizer_model == "llama":
        return SentencePieceTokenizer(metadata)
    if tokenizer_model == "gemma4":
        return Gemma4Tokenizer(metadata)
    if (
        tokenizer_model == "t5"
    ):  # SentencePiece Unigram: XLM-RoBERTa encoders (multilingual MiniLM, e5, bge-m3)
        return UnigramTokenizer(metadata)
    raise UnsupportedArchitectureError(f"Unsupported tokenizer.ggml.model: {tokenizer_model!r}")


def _has_byte_tokens(metadata: GGUFMetadata) -> bool:
    return _BYTE_TOKEN_TYPE in metadata.require("tokenizer.ggml.token_type")


_BYTE_TOKEN_TYPE = 6
# Every `tokenizer.ggml.model` build_tokenizer handles.
SUPPORTED_TOKENIZERS = frozenset({"gpt2", "bert", "llama", "gemma4", "t5"})
_supported_cache: dict[tuple[str, int, int], bool] = {}


def tokenizer_supported(gguf_path: str | Path) -> bool:
    """False when `gguf_path` names a tokenizer build_tokenizer can't build, so `GET /api/tags`
    can say so (`effective_capabilities`'s "tokenizer_unsupported") instead of listing a model that
    fails on its first request. True without a tokenizer (an mmproj, a forecaster) and when the file
    can't be read: one bad file must not take down the listing (as `unsupported_features`)."""
    path = Path(gguf_path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in _supported_cache:
            model = GGUFReader(path).read().metadata.get("tokenizer.ggml.model")
            _supported_cache[key] = model is None or model in SUPPORTED_TOKENIZERS
        return _supported_cache[key]
    except Exception:  # noqa: BLE001 - a listing must survive any unreadable file
        return True
