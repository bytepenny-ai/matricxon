"""Cross-checks Matricxon's TimesFM 2.5 forecasts (ForecastEngine) against HF transformers'
`TimesFm2_5ModelForPrediction` on the same series:

    .venv/bin/python -m tools.oracle.validate_timesfm25 MODEL.gguf [HF_REPO_OR_DIR]

MODEL.gguf is a `timesfm25` GGUF (amaye15/timesfm-gguf); the HF side defaults to
google/timesfm-2.5-200m-transformers. Compares every output (mean + 9 quantiles) per step,
relative to the series' scale. Runs on one thread to keep the laptop cool."""

import math
import sys
from types import SimpleNamespace

import torch

HORIZON = 64
# Max |difference| / series std, per weight type of the GGUF.
# Q8_0 measured 2026-11-10: up to 5.7e-2 (outer quantiles, linear series), median 1.9e-2.
TOLERANCE = {"F32": 1e-3, "F16": 1e-2, "Q8_0": 1e-1, "default": 1e-1}


class Series:
    """Test inputs: short and long, trend, seasonality, negatives, not patch-aligned."""

    @staticmethod
    def all() -> dict[str, list[float]]:
        gen = torch.Generator().manual_seed(0)
        noise = torch.randn(700, generator=gen).tolist()
        return {
            "short_7": [3.0, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5],
            "linear_100": [0.5 * i + 2 for i in range(100)],
            "seasonal_noisy_300": [
                10 + 3 * math.sin(2 * math.pi * i / 24) + 0.3 * noise[i] for i in range(300)
            ],
            "negative_walk_700": list(torch.tensor(noise).cumsum(0).tolist()),
        }


class Oracle:
    def __init__(self, gguf_path: str, hf_source: str) -> None:
        torch.set_num_threads(1)
        from transformers import TimesFm2_5ModelForPrediction

        from app.architectures.timesfm25 import TimesFm25Architecture
        from app.gguf.loader import GGUFModelLoader
        from app.runtime.forecast_engine import ForecastEngine

        loader = GGUFModelLoader(gguf_path, dtype=torch.float32)
        self._file_type = self._weight_type(loader)
        model = TimesFm25Architecture.from_gguf(loader)
        self._engine = ForecastEngine(SimpleNamespace(tag="oracle", architecture=model))
        self._hf = TimesFm2_5ModelForPrediction.from_pretrained(hf_source).eval()

    @staticmethod
    def _weight_type(loader: object) -> str:
        name = loader._tensor_infos_by_name["blk.0.attn_qkv.weight"].ggml_type  # noqa: SLF001
        return {0: "F32", 1: "F16", 8: "Q8_0"}.get(int(name), "default")

    def run(self) -> bool:
        tolerance = TOLERANCE[self._file_type]
        ok = True
        for name, values in Series.all().items():
            mx = self._engine.forecast([values], HORIZON)[0]
            mx_full = torch.tensor([mx.mean, *mx.quantiles.values()]).T
            with torch.no_grad():
                out = self._hf(past_values=[torch.tensor(values)])
            hf_full = out.full_predictions[0, :HORIZON]
            scale = max(float(torch.tensor(values).std()), 1e-6)
            err = float((mx_full - hf_full).abs().max()) / scale
            point_err = float(
                (torch.tensor(mx.point) - out.mean_predictions[0, :HORIZON]).abs().max()
            )
            passed = err <= tolerance
            ok = ok and passed
            print(
                f"{name:20s} max_rel_err={err:.2e} point_abs_err={point_err / scale:.2e} "
                f"{'ok' if passed else 'FAIL'}"
            )
        print(f"weights={self._file_type} tolerance={tolerance}")
        print("PASS" if ok else "FAIL")
        return ok


if __name__ == "__main__":
    hf = sys.argv[2] if len(sys.argv) > 2 else "google/timesfm-2.5-200m-transformers"
    sys.exit(0 if Oracle(sys.argv[1], hf).run() else 1)
