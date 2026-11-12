import base64
import io
import math

import numpy as np
import torch
from PIL import Image

# HF accepts 70, 140, 280, 560 or 1120 soft tokens per image (default 280); the smallest keeps the
# image prompt short on a CPU: at most 630 patches through the encoder, 70 tokens in the chat.
SOFT_TOKENS = 70
_POOL = 3


class Gemma4ImagePreprocessor:
    """Base64 image -> `(1, 3, H, W)` in [0, 1] for Gemma4VisionEncoder, as HF's
    `Gemma4ImageProcessorPil`: aspect ratio kept, resized (bicubic) to the largest size whose sides
    are multiples of `patch_size * 3` and whose patches fit `soft_tokens * 9`. No mean/std: the
    encoder maps [0, 1] to [-1, 1] itself."""

    def __init__(self, patch_size: int, soft_tokens: int = SOFT_TOKENS) -> None:
        self._patch = patch_size
        self._max_patches = soft_tokens * _POOL * _POOL

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """HF `get_aspect_ratio_preserving_size`."""
        side = _POOL * self._patch
        factor = math.sqrt(self._max_patches * self._patch**2 / (height * width))
        h = int(math.floor(factor * height / side)) * side
        w = int(math.floor(factor * width / side)) * side
        longest = (self._max_patches // _POOL**2) * side
        if h == 0 and w == 0:
            raise ValueError(f"A {width}x{height} image is too small to read.")
        if h == 0:
            h, w = side, min(int(math.floor(width / height)) * side, longest)
        elif w == 0:
            h, w = min(int(math.floor(height / width)) * side, longest), side
        return h, w

    def preprocess(self, image_base64: str) -> torch.Tensor:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        return self.from_image(image)

    def from_image(self, image: Image.Image) -> torch.Tensor:
        h, w = self.target_size(image.height, image.width)
        if (h, w) != (image.height, image.width):
            image = image.resize((w, h), Image.BICUBIC)
        pixels = torch.from_numpy(np.array(image)).to(torch.float32) / 255.0
        return pixels.permute(2, 0, 1).unsqueeze(0)
