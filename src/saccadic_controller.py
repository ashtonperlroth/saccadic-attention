"""Saccadic Controller: learned fixation policy using cross-attention + Gumbel-softmax."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaccadicController(nn.Module):
    """Decides where to fixate next based on peripheral map and current state.

    Computes cross-attention scores between the state query and peripheral map keys,
    then selects a block via Gumbel-softmax (training) or argmax (inference).

    Input:  peripheral_map (batch, num_blocks, D), state (batch, D), fixation_history
    Output: fixation_point (batch,), fixation_logits (batch, num_blocks)
    """

    def __init__(
        self,
        hidden_dim: int,
        block_size: int = 32,
        temperature: float = 1.0,
        mask_fixated: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.temperature = temperature
        self.mask_fixated = mask_fixated

        # Project state into query space
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        # Project peripheral map into key space
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        peripheral_map: torch.Tensor,
        state: torch.Tensor,
        fixation_history: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            peripheral_map: (batch, num_blocks, hidden_dim)
            state: (batch, hidden_dim)
            fixation_history: list of (batch,) tensors — previously selected block indices
            attention_mask: (batch, num_blocks) — 1 for valid blocks, 0 for padding

        Returns:
            fixation_point: (batch,) — token-level fixation positions
            fixation_logits: (batch, num_blocks) — raw scores (for visualization/entropy bonus)
        """
        batch, num_blocks, hidden_dim = peripheral_map.shape  # (B, M, D)

        # query: (batch, D)
        query = self.query_proj(state)
        # keys: (batch, num_blocks, D)
        keys = self.key_proj(peripheral_map)

        # scores: (batch, num_blocks) — dot product attention
        scores = torch.einsum('bd,bmd->bm', query, keys) / math.sqrt(hidden_dim)

        # Mask padded blocks
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Mask previously fixated blocks
        if self.mask_fixated and fixation_history:
            mask = torch.zeros(batch, num_blocks, device=scores.device, dtype=torch.bool)
            for prev_block_idx in fixation_history:
                # prev_block_idx: (batch,) — block indices
                mask.scatter_(1, prev_block_idx.unsqueeze(1), True)
            scores = scores.masked_fill(mask, float('-inf'))

        fixation_logits = scores  # (batch, num_blocks) — save for entropy bonus

        if self.training:
            # Gumbel-softmax with straight-through estimator
            # soft_selection: (batch, num_blocks) — approximately one-hot
            soft_selection = F.gumbel_softmax(scores, tau=self.temperature, hard=True)

            # Compute block index as weighted sum (differentiable via straight-through)
            block_indices_float = torch.arange(
                num_blocks, device=scores.device, dtype=torch.float
            )
            # selected_block: (batch,) — soft block index
            selected_block = torch.einsum('bm,m->b', soft_selection, block_indices_float)
        else:
            # Argmax during inference
            # selected_block: (batch,)
            selected_block = scores.argmax(dim=-1).float()

        # Convert block index to token position: (batch,)
        fixation_point = (selected_block * self.block_size).long()

        # Also return block indices for fixation history tracking
        block_idx = selected_block.long()

        return fixation_point, fixation_logits, block_idx
