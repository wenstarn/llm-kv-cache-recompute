# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional, Protocol

# Third Party
import torch


@dataclass
class LMCSelectionInputs:
    """Inputs for selecting which token positions stay active."""

    positions: torch.Tensor
    fresh_k: torch.Tensor
    fresh_v: torch.Tensor
    cached_k: torch.Tensor
    cached_v: torch.Tensor
    recompute_ratio: float
    chunk_boundary_positions: Optional[torch.Tensor] = None
    force_chunk_boundaries: bool = False


@dataclass
class LMCSelectionResult:
    """Selection result plus diagnostics for debugging experiments."""

    selected_indices: torch.Tensor
    topk_indices: torch.Tensor
    boundary_indices: torch.Tensor


class LMCSelectionStrategy(Protocol):
    """Protocol for token-selection experiments in CacheBlend."""

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        """Return selected indices plus debug information."""


class TopKDiffKSelector:
    """Select tokens with the largest squared K-vector deviation."""

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = _squared_diff(inputs.fresh_k, inputs.cached_k)
        return _build_selection_result(inputs, score)


class TopKDiffVSelector:
    """Select tokens with the largest squared V-vector deviation."""

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = _squared_diff(inputs.fresh_v, inputs.cached_v)
        return _build_selection_result(inputs, score)


class TopKDiffKVSumSelector:
    """Select tokens by the sum of squared K and V deviations."""

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = _squared_diff(inputs.fresh_k, inputs.cached_k) + _squared_diff(
            inputs.fresh_v, inputs.cached_v
        )
        return _build_selection_result(inputs, score)


class TopKDiffKVWeightedSelector:
    """Select tokens by a weighted combination of K and V deviations."""

    def __init__(self, k_weight: float, v_weight: float) -> None:
        self.k_weight = k_weight
        self.v_weight = v_weight

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = self.k_weight * _squared_diff(
            inputs.fresh_k, inputs.cached_k
        ) + self.v_weight * _squared_diff(inputs.fresh_v, inputs.cached_v)
        return _build_selection_result(inputs, score)


class TopKDiffKNeighborSelector:
    """Expand around the highest K-deviation tokens under a final token budget."""

    def __init__(
        self,
        final_recompute_ratio: float,
        expand_top_ratio: float,
        expand_radius: int,
        expand_direction: str,
    ) -> None:
        self.final_recompute_ratio = final_recompute_ratio
        self.expand_top_ratio = expand_top_ratio
        self.expand_radius = expand_radius
        self.expand_direction = expand_direction

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = _squared_diff(inputs.fresh_k, inputs.cached_k)
        return _build_neighbor_selection_result(
            inputs,
            score,
            final_recompute_ratio=self.final_recompute_ratio,
            expand_top_ratio=self.expand_top_ratio,
            expand_radius=self.expand_radius,
            expand_direction=self.expand_direction,
        )


class TopKDiffKSpanClosingSelector:
    """Fill short gaps between high K-deviation tokens under a final budget."""

    def __init__(self, final_recompute_ratio: float, max_gap: int) -> None:
        self.final_recompute_ratio = final_recompute_ratio
        self.max_gap = max_gap

    def select(self, inputs: LMCSelectionInputs) -> LMCSelectionResult:
        score = _squared_diff(inputs.fresh_k, inputs.cached_k)
        return _build_span_closing_selection_result(
            inputs,
            score,
            final_recompute_ratio=self.final_recompute_ratio,
            max_gap=self.max_gap,
        )


def _squared_diff(fresh: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
    return torch.sum((fresh.to(torch.float32) - cached.to(torch.float32)) ** 2, dim=[1])


def _get_top_indices(
    score: torch.Tensor,
    recompute_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_len = score.shape[0]
    topk_num = int(total_len * recompute_ratio)
    topk_num = max(topk_num, 1)

    ranked_top_indices = torch.topk(score, k=topk_num).indices
    sorted_top_indices, _ = torch.sort(ranked_top_indices)
    return ranked_top_indices, sorted_top_indices


def _build_selection_result(
    inputs: LMCSelectionInputs,
    score: torch.Tensor,
) -> LMCSelectionResult:
    _, top_indices = _get_top_indices(score, inputs.recompute_ratio)
    boundary_indices = torch.empty(0, dtype=torch.long, device=top_indices.device)

    if (
        inputs.force_chunk_boundaries
        and inputs.chunk_boundary_positions is not None
        and inputs.chunk_boundary_positions.numel() > 0
    ):
        boundary_mask = torch.isin(
            inputs.positions,
            inputs.chunk_boundary_positions.to(
                device=inputs.positions.device,
                dtype=inputs.positions.dtype,
            ),
        )
        boundary_indices = torch.nonzero(boundary_mask, as_tuple=True)[0]
        if boundary_indices.numel() > 0:
            selected_indices = torch.unique(
                torch.cat([top_indices, boundary_indices]), sorted=True
            )
            return LMCSelectionResult(
                selected_indices=selected_indices,
                topk_indices=top_indices,
                boundary_indices=boundary_indices,
            )

    return LMCSelectionResult(
        selected_indices=top_indices,
        topk_indices=top_indices,
        boundary_indices=boundary_indices,
    )


def _get_budget_num(
    total_len: int,
    base_count: int,
    final_recompute_ratio: float,
) -> int:
    if final_recompute_ratio <= 0:
        return total_len
    budget_num = int(total_len * final_recompute_ratio)
    budget_num = max(budget_num, base_count)
    return min(budget_num, total_len)


def _apply_budget(
    base_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    budget_num: int,
) -> torch.Tensor:
    selected_indices = torch.unique(base_indices, sorted=True)
    if candidate_indices.numel() == 0 or selected_indices.numel() >= budget_num:
        return selected_indices

    candidate_indices = torch.unique(candidate_indices, sorted=False)
    candidate_indices = candidate_indices[
        ~torch.isin(candidate_indices, selected_indices)
    ]
    if candidate_indices.numel() == 0:
        return selected_indices

    remaining_budget = budget_num - selected_indices.numel()
    ranked_candidate_positions = torch.topk(
        candidate_scores[candidate_indices],
        k=min(remaining_budget, candidate_indices.numel()),
    ).indices
    kept_candidates = candidate_indices[ranked_candidate_positions]
    return torch.unique(torch.cat([selected_indices, kept_candidates]), sorted=True)


def _build_neighbor_selection_result(
    inputs: LMCSelectionInputs,
    score: torch.Tensor,
    final_recompute_ratio: float,
    expand_top_ratio: float,
    expand_radius: int,
    expand_direction: str,
) -> LMCSelectionResult:
    ranked_top_indices, top_indices = _get_top_indices(score, inputs.recompute_ratio)
    boundary_indices = torch.empty(0, dtype=torch.long, device=top_indices.device)
    total_len = score.shape[0]
    budget_num = _get_budget_num(total_len, len(top_indices), final_recompute_ratio)
    expand_radius = max(expand_radius, 0)
    if expand_radius == 0:
        return LMCSelectionResult(
            selected_indices=top_indices,
            topk_indices=top_indices,
            boundary_indices=boundary_indices,
        )

    anchor_count = int(len(ranked_top_indices) * expand_top_ratio)
    anchor_count = max(anchor_count, 1)
    anchor_count = min(anchor_count, len(ranked_top_indices))
    anchors = ranked_top_indices[:anchor_count]

    offsets: list[int] = []
    if expand_direction in ("left", "both"):
        offsets.extend(range(-expand_radius, 0))
    if expand_direction in ("right", "both"):
        offsets.extend(range(1, expand_radius + 1))
    if not offsets:
        raise ValueError(f"Invalid blend_neighbor_expand_direction: {expand_direction}")

    candidate_chunks = []
    for offset in offsets:
        candidates = anchors + offset
        candidates = candidates[(candidates >= 0) & (candidates < total_len)]
        if candidates.numel() > 0:
            candidate_chunks.append(candidates)

    if not candidate_chunks:
        selected_indices = top_indices
    else:
        candidate_indices = torch.cat(candidate_chunks)
        selected_indices = _apply_budget(
            top_indices,
            candidate_indices,
            score,
            budget_num,
        )

    return LMCSelectionResult(
        selected_indices=selected_indices,
        topk_indices=top_indices,
        boundary_indices=boundary_indices,
    )


def _build_span_closing_selection_result(
    inputs: LMCSelectionInputs,
    score: torch.Tensor,
    final_recompute_ratio: float,
    max_gap: int,
) -> LMCSelectionResult:
    _, top_indices = _get_top_indices(score, inputs.recompute_ratio)
    boundary_indices = torch.empty(0, dtype=torch.long, device=top_indices.device)
    budget_num = _get_budget_num(
        score.shape[0],
        len(top_indices),
        final_recompute_ratio,
    )
    max_gap = max(max_gap, 0)
    if max_gap == 0 or len(top_indices) < 2:
        return LMCSelectionResult(
            selected_indices=top_indices,
            topk_indices=top_indices,
            boundary_indices=boundary_indices,
        )

    candidate_chunks = []
    previous_idx = int(top_indices[0].item())
    for current in top_indices[1:]:
        current_idx = int(current.item())
        gap = current_idx - previous_idx - 1
        if 0 < gap <= max_gap:
            candidate_chunks.append(
                torch.arange(
                    previous_idx + 1,
                    current_idx,
                    device=top_indices.device,
                    dtype=top_indices.dtype,
                )
            )
        previous_idx = current_idx

    if not candidate_chunks:
        selected_indices = top_indices
    else:
        candidate_indices = torch.cat(candidate_chunks)
        selected_indices = _apply_budget(
            top_indices,
            candidate_indices,
            score,
            budget_num,
        )

    return LMCSelectionResult(
        selected_indices=selected_indices,
        topk_indices=top_indices,
        boundary_indices=boundary_indices,
    )


def build_selection_strategy(
    strategy_name: str,
    k_weight: float,
    v_weight: float,
    final_recompute_ratio: float = 0.0,
    expand_top_ratio: float = 1.0,
    expand_radius: int = 1,
    expand_direction: str = "both",
    span_close_max_gap: int = 1,
) -> LMCSelectionStrategy:
    """Construct a token-selection strategy by name."""

    if strategy_name == "topk_diff_k":
        return TopKDiffKSelector()
    if strategy_name == "topk_diff_v":
        return TopKDiffVSelector()
    if strategy_name == "topk_diff_kv_sum":
        return TopKDiffKVSumSelector()
    if strategy_name == "topk_diff_kv_weighted":
        return TopKDiffKVWeightedSelector(k_weight=k_weight, v_weight=v_weight)
    if strategy_name == "topk_diff_k_neighbors":
        return TopKDiffKNeighborSelector(
            final_recompute_ratio=final_recompute_ratio,
            expand_top_ratio=expand_top_ratio,
            expand_radius=expand_radius,
            expand_direction=expand_direction,
        )
    if strategy_name == "topk_diff_k_span_closing":
        return TopKDiffKSpanClosingSelector(
            final_recompute_ratio=final_recompute_ratio,
            max_gap=span_close_max_gap,
        )
    raise ValueError(f"Unknown blend selection strategy: {strategy_name}")
