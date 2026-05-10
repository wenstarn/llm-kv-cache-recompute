# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Protocol

# Third Party
import torch


@dataclass
class LMCFusionInputs:
    """Inputs for fusing recomputed K/V with cached K/V."""

    selected_indices: torch.Tensor
    selected_k: torch.Tensor
    selected_v: torch.Tensor
    cached_k: torch.Tensor
    cached_v: torch.Tensor


class LMCFusionStrategy(Protocol):
    """Protocol for KV-fusion experiments in CacheBlend."""

    def fuse(self, inputs: LMCFusionInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full fused K/V tensors."""


class OverwriteSelectedFusion:
    """Replace cached K/V at selected positions with recomputed K/V."""

    def fuse(self, inputs: LMCFusionInputs) -> tuple[torch.Tensor, torch.Tensor]:
        fused_k = inputs.cached_k
        fused_v = inputs.cached_v
        fused_k[inputs.selected_indices] = inputs.selected_k
        fused_v[inputs.selected_indices] = inputs.selected_v
        return fused_k, fused_v


class WeightedSelectedFusion:
    """Blend cached and recomputed K/V at selected positions."""

    def __init__(self, cached_k_weight: float, cached_v_weight: float) -> None:
        self.cached_k_weight = cached_k_weight
        self.cached_v_weight = cached_v_weight

    def fuse(self, inputs: LMCFusionInputs) -> tuple[torch.Tensor, torch.Tensor]:
        fused_k = inputs.cached_k
        fused_v = inputs.cached_v

        selected_cached_k = fused_k[inputs.selected_indices]
        selected_cached_v = fused_v[inputs.selected_indices]

        fused_k[inputs.selected_indices] = (
            self.cached_k_weight * selected_cached_k
            + (1.0 - self.cached_k_weight) * inputs.selected_k
        )
        fused_v[inputs.selected_indices] = (
            self.cached_v_weight * selected_cached_v
            + (1.0 - self.cached_v_weight) * inputs.selected_v
        )
        return fused_k, fused_v


class AdaptiveWeightedSelectedFusion:
    """Blend cached and recomputed K/V with token-wise adaptive weights."""

    def __init__(
        self,
        min_cached_k_weight: float,
        max_cached_k_weight: float,
        min_cached_v_weight: float,
        max_cached_v_weight: float,
        eps: float = 1e-6,
    ) -> None:
        self.min_cached_k_weight = min_cached_k_weight
        self.max_cached_k_weight = max_cached_k_weight
        self.min_cached_v_weight = min_cached_v_weight
        self.max_cached_v_weight = max_cached_v_weight
        self.eps = eps

    def fuse(self, inputs: LMCFusionInputs) -> tuple[torch.Tensor, torch.Tensor]:
        fused_k = inputs.cached_k
        fused_v = inputs.cached_v

        selected_cached_k = fused_k[inputs.selected_indices]
        selected_cached_v = fused_v[inputs.selected_indices]

        k_weights = self._compute_cached_weights(
            inputs.selected_k,
            selected_cached_k,
            self.min_cached_k_weight,
            self.max_cached_k_weight,
        )
        v_weights = self._compute_cached_weights(
            inputs.selected_v,
            selected_cached_v,
            self.min_cached_v_weight,
            self.max_cached_v_weight,
        )

        fused_k[inputs.selected_indices] = (
            k_weights * selected_cached_k + (1.0 - k_weights) * inputs.selected_k
        )
        fused_v[inputs.selected_indices] = (
            v_weights * selected_cached_v + (1.0 - v_weights) * inputs.selected_v
        )
        return fused_k, fused_v

    def _compute_cached_weights(
        self,
        recomputed: torch.Tensor,
        cached: torch.Tensor,
        min_weight: float,
        max_weight: float,
    ) -> torch.Tensor:
        diff = torch.sum((recomputed - cached).to(torch.float32) ** 2, dim=1)
        if diff.numel() == 0:
            return torch.empty_like(diff, dtype=torch.float32)

        diff_min = diff.min()
        diff_max = diff.max()
        norm = (diff - diff_min) / (diff_max - diff_min + self.eps)

        cached_weights = max_weight - norm * (max_weight - min_weight)
        cached_weights = torch.clamp(cached_weights, min=min_weight, max=max_weight)
        return cached_weights.to(device=recomputed.device, dtype=recomputed.dtype).view(
            -1, 1
        )


def build_fusion_strategy(
    strategy_name: str,
    cached_k_weight: float,
    cached_v_weight: float,
    min_cached_k_weight: float,
    max_cached_k_weight: float,
    min_cached_v_weight: float,
    max_cached_v_weight: float,
) -> LMCFusionStrategy:
    """Construct a KV-fusion strategy by name."""

    if strategy_name == "overwrite_selected":
        return OverwriteSelectedFusion()
    if strategy_name == "weighted_selected":
        return WeightedSelectedFusion(cached_k_weight, cached_v_weight)
    if strategy_name == "adaptive_weighted_selected":
        return AdaptiveWeightedSelectedFusion(
            min_cached_k_weight=min_cached_k_weight,
            max_cached_k_weight=max_cached_k_weight,
            min_cached_v_weight=min_cached_v_weight,
            max_cached_v_weight=max_cached_v_weight,
        )
    raise ValueError(f"Unknown blend fusion strategy: {strategy_name}")
