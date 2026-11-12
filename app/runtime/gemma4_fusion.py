"""Gemma 4 image fusion: the counterpart of `vision_fusion.build_prompt_with_images` for a `gemma4v`
mmproj. Each `[IMG]` marker (`ChatTemplatePromptBuilder`, where the template would write
`<|image|>`) becomes `<|image>` + one placeholder per soft token + `<image|>`, as HF's
`Gemma4Processor` expands it, and the image goes through `Gemma4VisionEncoder`.
`Gemma4Architecture._forward_impl` overwrites the placeholders' embedding with the image's soft
tokens and reads them as the pad token for the per-layer lookup (HF swaps the image token for pad
there). Positions stay plain 1D (Gemma 4 has no M-RoPE).

The placeholder ids are derived from the image's bytes, so the same picture always gives the same
ids and two pictures never share them: the prompt cache can then reuse a chat with an image (the
next message, even "OK", needn't re-read it), which identical placeholders made unsafe. Encoded
images are kept by the same hash, so a picture sent again with later messages isn't re-encoded."""

import hashlib
from collections import OrderedDict

import torch

from app.gguf.loader import GGUFModelLoader
from app.models.installed_model import InstalledModel
from app.runtime.vision_fusion import IMAGE_MARKER
from app.vision.device import VisionDevice
from app.vision.gemma4v_encoder import Gemma4VisionEncoder
from app.vision.gemma4v_preprocessing import Gemma4ImagePreprocessor

_IMAGE_START, _IMAGE_END = "<|image>", "<image|>"
# Placeholder ids are drawn from this range of ordinary vocab entries (their own embedding is never
# read, see the module docstring).
_ID_BASE, _ID_SPAN = 1000, 200_000
_KEPT_IMAGES = 16
_encoder_cache: dict[tuple[str, str], Gemma4VisionEncoder] = {}
_soft_tokens: OrderedDict[str, torch.Tensor] = OrderedDict()


def placeholder_ids(image_base64: str, count: int) -> list[int]:
    """`count` ids that depend only on the image."""
    seed = int.from_bytes(hashlib.sha256(image_base64.encode()).digest()[:8], "big")
    return [_ID_BASE + (seed + k * 7919) % _ID_SPAN for k in range(count)]


def _get_encoder(mmproj: InstalledModel, device: torch.device) -> Gemma4VisionEncoder:
    key = (mmproj.tag, str(device))
    if key not in _encoder_cache:
        loader = GGUFModelLoader(mmproj.path, dtype=torch.float32)
        _encoder_cache[key] = VisionDevice.place(Gemma4VisionEncoder.from_gguf(loader), device)
    return _encoder_cache[key]


def build_gemma4_prompt(
    prompt: str,
    images_b64: list[str],
    tokenizer: object,
    mmproj: InstalledModel,
    add_bos: bool,
    device: torch.device | None = None,
) -> tuple[list[int], list[tuple[int, torch.Tensor]]]:
    """One `[IMG]` marker per image, in order (a marker without an image is dropped). Returns the
    token ids and `(start, soft tokens)` spans for `Gemma4Architecture._forward_impl`."""
    encoder = _get_encoder(mmproj, device or torch.device("cpu"))
    preprocessor = Gemma4ImagePreprocessor(encoder.patch_size)
    segments = prompt.split(IMAGE_MARKER)
    token_ids: list[int] = []
    embeddings: list[tuple[int, torch.Tensor]] = []
    with torch.no_grad():
        for i, segment in enumerate(segments):
            has_image = i < len(segments) - 1 and i < len(images_b64)
            text = ("" if i == 0 or i > len(images_b64) else _IMAGE_END) + segment
            token_ids.extend(
                tokenizer.encode(text + (_IMAGE_START if has_image else ""), add_bos and i == 0)
            )
            if not has_image:
                continue
            soft_tokens = _encoded(encoder, preprocessor, mmproj.tag, images_b64[i])
            embeddings.append((len(token_ids), soft_tokens))
            token_ids.extend(placeholder_ids(images_b64[i], soft_tokens.shape[0]))
    return token_ids, embeddings


def _encoded(
    encoder: Gemma4VisionEncoder, preprocessor: Gemma4ImagePreprocessor, tag: str, image: str
) -> torch.Tensor:
    """The image's soft tokens, kept for the latest `_KEPT_IMAGES` pictures."""
    key = tag + hashlib.sha256(image.encode()).hexdigest()
    if key in _soft_tokens:
        _soft_tokens.move_to_end(key)
        return _soft_tokens[key]
    soft_tokens = VisionDevice.encode(encoder, preprocessor.preprocess(image))
    _soft_tokens[key] = soft_tokens
    if len(_soft_tokens) > _KEPT_IMAGES:
        _soft_tokens.popitem(last=False)
    return soft_tokens
