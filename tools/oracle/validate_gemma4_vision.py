"""Cross-checks Gemma4VisionEncoder + Gemma4ImagePreprocessor (gemma4v mmproj) against HF's
Gemma4ImageProcessorPil, Gemma4VisionModel and Gemma4MultimodalEmbedder on the original weights.

Run in a memory cap (~1.5 GB used):
  systemd-run --user --scope -p MemoryMax=3G -- .venv/bin/python \
      tools/oracle/validate_gemma4_vision.py <mmproj.gguf> <vision.safetensors> <config.json>

`vision.safetensors` holds only the `model.vision_tower.*` and `model.embed_vision.*` tensors of
`unsloth/gemma-4-E2B-it`'s model.safetensors (fetched by byte range; the full file is 10 GB)."""

import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

from app.gguf.loader import GGUFModelLoader
from app.vision.gemma4v_encoder import Gemma4VisionEncoder
from app.vision.gemma4v_preprocessing import SOFT_TOKENS, Gemma4ImagePreprocessor


def _test_image(width: int, height: int) -> Image.Image:
    """A synthetic picture with gradients, shapes and edges (no real photo needed)."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    rgb = np.stack([np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)),
                    (x + y) % 256], axis=-1).astype(np.uint8)  # fmt: skip
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.ellipse((width * 0.2, height * 0.2, width * 0.6, height * 0.7), fill=(30, 90, 200))
    draw.rectangle((width * 0.55, height * 0.1, width * 0.9, height * 0.4), fill=(240, 220, 40))
    draw.line((0, height - 1, width - 1, 0), fill=(10, 10, 10), width=5)
    return image


def _part(weights: dict, prefix: str) -> dict:
    return {k.removeprefix(prefix): v.float() for k, v in weights.items() if k.startswith(prefix)}


def _reference(weights: dict, config: dict) -> tuple[Gemma4VisionModel, Gemma4MultimodalEmbedder]:
    vision_config = Gemma4VisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "eager"
    text_config = Gemma4TextConfig(**config["text_config"])
    tower, embedder = (
        Gemma4VisionModel(vision_config),
        Gemma4MultimodalEmbedder(vision_config, text_config),
    )
    tower.load_state_dict(_part(weights, "model.vision_tower."))
    embedder.load_state_dict(_part(weights, "model.embed_vision."))
    return tower.float().eval(), embedder.float().eval()


def main() -> int:
    mmproj, safetensors_path, config_path = sys.argv[1:4]
    config = json.load(open(config_path))
    tower, embedder = _reference(load_file(safetensors_path), config)
    ours = Gemma4VisionEncoder.from_gguf(GGUFModelLoader(mmproj, dtype=torch.float32))
    processor = Gemma4ImageProcessorPil(max_soft_tokens=SOFT_TOKENS)
    preprocessor = Gemma4ImagePreprocessor(ours.patch_size)
    failed = 0
    for width, height in ((640, 480), (300, 900), (1024, 1024)):
        image = _test_image(width, height)
        hf_in = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            hf_tokens = tower(
                hf_in["pixel_values"].float(), hf_in["image_position_ids"]
            ).last_hidden_state
            hf = embedder(hf_tokens.unsqueeze(0))[0]
            pixels = preprocessor.from_image(image)
            mine = ours(pixels)
        same_shape = hf.shape == mine.shape
        cosine = (
            torch.nn.functional.cosine_similarity(hf, mine, dim=-1)
            if same_shape
            else torch.zeros(1)
        )
        rel = (hf - mine).norm() / hf.norm() if same_shape else torch.tensor(float("inf"))
        ok = same_shape and cosine.min() > 0.99 and rel < 0.05
        failed += not ok
        verdict = "PASS" if ok else "FAIL"
        print(f"[{verdict}] {width}x{height} -> {tuple(pixels.shape[-2:])}:")
        print(f"  tokens hf {tuple(hf.shape)} ours {tuple(mine.shape)}")
        print(f"  cosine min {cosine.min():.4f}, rel err {rel:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
