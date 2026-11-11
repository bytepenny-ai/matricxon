"""ForecastEngine against HF transformers' TimesFm2_5ModelForPrediction, both built from the same
tiny random `timesfm25` GGUF: catches a wrong tensor mapping or a skipped step without a 900 MB
download (tools/oracle/validate_timesfm25.py does that on the real weights)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.architectures.timesfm25 import TimesFm25Architecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.forecast_engine import ForecastEngine
from tests import tiny_gguf_timesfm25 as tiny

transformers = pytest.importorskip("transformers")


class HfTwin:
    """The HF model with the GGUF's weights copied in under HF's names."""

    _BLOCKS = {
        "input_layer": "hidden",
        "output_layer": "output",
        "residual_layer": "skip",
    }
    _LAYER = {
        "input_layernorm.weight": "pre_attn_norm.weight",
        "post_attention_layernorm.weight": "post_attn_norm.weight",
        "pre_feedforward_layernorm.weight": "pre_ff_norm.weight",
        "post_feedforward_layernorm.weight": "post_ff_norm.weight",
        "self_attn.o_proj.weight": "attn_out.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.scaling": "attn_q_scale.weight",
        "mlp.fc1.weight": "ffn_up.weight",
        "mlp.fc2.weight": "ffn_down.weight",
    }

    @classmethod
    def build(cls, gguf_path: Path) -> torch.nn.Module:
        config = transformers.TimesFm2_5Config(
            patch_length=tiny.PATCH,
            context_length=64,
            horizon_length=tiny.HORIZON,
            output_quantile_len=tiny.QUANTILE_LEN,
            num_hidden_layers=tiny.N_LAYER,
            hidden_size=tiny.N_EMBD,
            intermediate_size=tiny.N_EMBD,
            head_dim=tiny.HEAD_DIM,
            num_attention_heads=tiny.N_HEAD,
            num_key_value_heads=tiny.N_HEAD,
        )
        config._attn_implementation = "eager"
        model = transformers.TimesFm2_5ModelForPrediction(config).eval()
        state = model.state_dict()
        with GGUFModelLoader(gguf_path, dtype=torch.float32) as loader:
            for name, value in cls._mapping(loader).items():
                state[name].copy_(value)
        return model

    @classmethod
    def _mapping(cls, loader: GGUFModelLoader) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for hf_block, ours, bias in (
            ("model.input_ff_layer", "tokenizer", True),
            ("output_projection_point", "out_point", False),
            ("output_projection_quantiles", "out_quantile", False),
        ):
            for hf_part, part in cls._BLOCKS.items():
                out[f"{hf_block}.{hf_part}.weight"] = loader.load_tensor(f"{ours}.{part}.weight")
                if bias:
                    out[f"{hf_block}.{hf_part}.bias"] = loader.load_tensor(f"{ours}.{part}.bias")
        for i in range(tiny.N_LAYER):
            prefix = f"model.layers.{i}."
            for hf_name, name in cls._LAYER.items():
                out[prefix + hf_name] = loader.load_tensor(f"blk.{i}.{name}")
            q, k, v = loader.load_tensor(f"blk.{i}.attn_qkv.weight").chunk(3, dim=0)
            out[prefix + "self_attn.q_proj.weight"] = q
            out[prefix + "self_attn.k_proj.weight"] = k
            out[prefix + "self_attn.v_proj.weight"] = v
        return out


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[ForecastEngine, torch.nn.Module]:
    path = tiny.build_tiny_timesfm25_gguf(tmp_path_factory.mktemp("tfm") / "tiny.gguf", seed=3)
    model = TimesFm25Architecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32)).eval()
    return ForecastEngine(SimpleNamespace(tag="tiny", architecture=model)), HfTwin.build(path)


@pytest.mark.parametrize(
    "series",
    [
        [2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0],
        [0.5 * i - 3.0 for i in range(23)],
        [float((i * 7) % 5) for i in range(70)],
    ],
)
def test_matches_hf_transformers(
    pair: tuple[ForecastEngine, torch.nn.Module], series: list[float]
) -> None:
    engine, hf = pair
    ours = engine.forecast([series], tiny.HORIZON)[0]
    with torch.no_grad():
        theirs = hf(past_values=[torch.tensor(series)]).full_predictions[0]
    ours_full = torch.tensor([ours.mean, *ours.quantiles.values()]).T
    assert torch.allclose(ours_full, theirs, atol=1e-4, rtol=1e-4)
