"""Gemma 4 image support: the `gemma4v` encoder on a tiny random tower, HF's resize rule, the
prompt expansion (`<|image>` + one pad per soft token + `<image|>`) and the splice into
Gemma4Architecture (image rows in the main input, which the per-layer term reads as well).
The real tower is checked against HF in tools/oracle/validate_gemma4_vision.py."""

import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime import gemma4_fusion
from app.vision.gemma4v_encoder import Gemma4VisionEncoder
from app.vision.gemma4v_preprocessing import Gemma4ImagePreprocessor
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_ple_kv_shared_gguf


def _tiny_encoder(out_dim: int = 12) -> Gemma4VisionEncoder:
    meta = GGUFMetadata(
        {
            "general.architecture": "clip",
            "clip.vision.projector_type": "gemma4v",
            "clip.vision.embedding_length": 16,
            "clip.vision.attention.head_count": 2,
            "clip.vision.block_count": 2,
            "clip.vision.feed_forward_length": 32,
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.patch_size": 4,
        }
    )
    torch.manual_seed(0)
    encoder = Gemma4VisionEncoder(meta, out_dim)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        encoder.position_embd = torch.nn.Parameter(torch.randn(2, 64, 16) * 0.1)
    return encoder.eval()


def test_the_tower_pools_3x3_patches_into_text_width_tokens() -> None:
    encoder = _tiny_encoder()
    with torch.no_grad():
        out = encoder(torch.rand(1, 3, 24, 36))  # 6x9 patches of 4px -> 2x3 soft tokens
    assert out.shape == (6, 12) and torch.isfinite(out).all()
    assert encoder.soft_grid(24, 36) == (2, 3)


def test_clamped_linears_bound_their_input_and_output() -> None:
    encoder = _tiny_encoder()
    linear = encoder.layers[0].q
    linear.bounds.copy_(torch.tensor([-0.1, 0.1, -0.05, 0.05]))
    out = linear(torch.full((1, 16), 5.0))
    assert out.abs().max() <= 0.05


def test_rotary_at_position_zero_changes_nothing() -> None:
    encoder = _tiny_encoder()
    cos, sin = encoder._rotary(1, 1)
    x = torch.randn(1, 2, 8)
    assert torch.allclose(Gemma4VisionEncoder.rotate(x, cos, sin), x)


def test_images_are_resized_as_hf_does() -> None:
    # Values from HF's get_aspect_ratio_preserving_size (patch 16, pool 3, 70 soft tokens).
    pre = Gemma4ImagePreprocessor(16)
    assert pre.target_size(480, 640) == (336, 432)
    assert pre.target_size(900, 300) == (672, 192)
    assert pre.target_size(1024, 1024) == (384, 384)


def test_the_prompt_gets_markers_and_one_pad_per_soft_token(monkeypatch) -> None:
    class Tokenizer:
        def encode(self, text: str, add_bos: bool = False) -> list[int]:
            return ([2] if add_bos else []) + [ord(c) for c in text]

    class Encoder:
        patch_size = 16

    class Mmproj:
        tag = "mmproj"

    encodes: list[int] = []

    def encode(*_args) -> torch.Tensor:
        encodes.append(1)
        return torch.ones(3, 5)

    monkeypatch.setattr(gemma4_fusion, "_get_encoder", lambda *_args: Encoder())
    monkeypatch.setattr(gemma4_fusion.Gemma4ImagePreprocessor, "preprocess", lambda *_a: None)
    monkeypatch.setattr(gemma4_fusion.VisionDevice, "encode", encode)
    build = gemma4_fusion.build_gemma4_prompt
    ids, spans = build("A[IMG]B", ["img"], Tokenizer(), Mmproj(), True)
    text = [2, ord("A"), *map(ord, "<|image>")]
    placeholders = gemma4_fusion.placeholder_ids("img", 3)
    assert ids == [*text, *placeholders, *map(ord, "<image|>B")]
    assert spans[0][0] == len(text) and spans[0][1].shape == (3, 5)
    # Same picture, same ids (the prompt cache can match), encoded once; another picture differs.
    assert build("A[IMG]B", ["img"], Tokenizer(), Mmproj(), True)[0] == ids and len(encodes) == 1
    assert gemma4_fusion.placeholder_ids("other", 3) != placeholders


def test_image_rows_replace_the_input_the_per_layer_term_reads_too(tmp_path) -> None:
    model = Gemma4Architecture.from_gguf(
        GGUFModelLoader(build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "g.gguf"), torch.float32),
        dtype=torch.float32,
    )
    seen: list[torch.Tensor] = []
    seen_ids: list[torch.Tensor] = []
    original = model.per_layer_embedding.forward

    def capture(input_ids: torch.Tensor, scaled: torch.Tensor) -> torch.Tensor:
        seen.append(scaled.clone())
        seen_ids.append(input_ids.clone())
        return original(input_ids, scaled)

    model.per_layer_embedding.forward = capture
    ids = torch.tensor([[4, 30, 31, 20]])  # 30, 31: an image's placeholder ids
    image = torch.randn(2, model.n_embd)
    with torch.no_grad():
        plain = model._forward_impl(ids)
        fused = model._forward_impl(ids, image_embeddings=[(1, image)])
    assert not torch.allclose(plain, fused)
    # HF: the per-layer context part is projected from the final inputs, image rows included.
    assert torch.allclose(seen[1][0, 1:3], image)
    assert torch.allclose(seen[1][0, 0], seen[0][0, 0])  # text positions unchanged
    assert seen_ids[1].tolist() == [[4, 0, 0, 20]]  # placeholders read as <pad> per layer
