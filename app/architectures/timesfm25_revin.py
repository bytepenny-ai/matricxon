"""TimesFM's per-patch reversible instance normalization (RevIN) with running statistics."""

import torch

TOLERANCE = 1e-6


class RunningPatchStats:
    """Mean and std of every value up to and including each patch, padded values skipped
    (Welford's merge, as the reference computes it)."""

    @staticmethod
    def compute(patches: torch.Tensor, padded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`patches` (B, N, P), `padded` (B, N, P) bool -> mu, sigma (B, N)."""
        batch = patches.shape[0]
        count = patches.new_zeros(batch)
        mean = patches.new_zeros(batch)
        std = patches.new_zeros(batch)
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        for i in range(patches.shape[1]):
            count, mean, std = RunningPatchStats._merge(
                count, mean, std, patches[:, i], padded[:, i]
            )
            means.append(mean)
            stds.append(std)
        return torch.stack(means, dim=1), torch.stack(stds, dim=1)

    @staticmethod
    def _merge(
        count: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        values: torch.Tensor,
        padded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = (~padded).to(values.dtype)
        inc_count = valid.sum(dim=-1)
        safe_inc = inc_count.clamp(min=1)
        inc_mean = (values * valid).sum(dim=-1) / safe_inc
        inc_var = (((values - inc_mean[:, None]) * valid) ** 2).sum(dim=-1) / safe_inc
        new_count = count + inc_count
        safe_new = new_count.clamp(min=1)
        new_mean = (count * mean + inc_mean * inc_count) / safe_new
        new_var = (
            count * std**2
            + inc_count * inc_var
            + count * (mean - new_mean) ** 2
            + inc_count * (inc_mean - new_mean) ** 2
        ) / safe_new
        return new_count, new_mean, new_var.clamp(min=0.0).sqrt()


class RevIN:
    """Normalize with (mu, sigma), or reverse it; a near-zero sigma divides by 1."""

    @staticmethod
    def normalize(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        mu, sigma = RevIN._expand(x, mu), RevIN._expand(x, sigma)
        safe = torch.where(sigma < TOLERANCE, torch.ones_like(sigma), sigma)
        return (x - mu) / safe

    @staticmethod
    def denormalize(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x * RevIN._expand(x, sigma) + RevIN._expand(x, mu)

    @staticmethod
    def _expand(x: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        while stat.dim() < x.dim():
            stat = stat[..., None]
        return stat
