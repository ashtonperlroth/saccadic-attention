"""Saccadic Attention: combined module with iterative fixation loop."""

import torch
import torch.nn as nn

from src.peripheral_encoder import PeripheralEncoder
from src.foveal_processor import FovealProcessor
from src.saccadic_controller import SaccadicController


class SaccadicAttention(nn.Module):
    """Full saccadic attention module combining peripheral encoder, foveal processor,
    and saccadic controller in an iterative fixation loop.

    Forward pass:
        1. Peripheral encoding (one-time, O(n))
        2. Iterative saccadic loop: controller selects fixation -> foveal processor updates state
        3. Output projection

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, hidden_dim), fixation_info dict
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_saccades: int = 5,
        window_size: int = 128,
        block_size: int = 32,
        gumbel_temperature: float = 1.0,
        mask_fixated: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_saccades = num_saccades
        self.window_size = window_size
        self.block_size = block_size

        # Component 1: Peripheral Encoder
        self.peripheral_encoder = PeripheralEncoder(
            hidden_dim=hidden_dim,
            block_size=block_size,
        )

        # Component 2: Foveal Processor
        self.foveal_processor = FovealProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
        )

        # Component 3: Saccadic Controller
        self.saccadic_controller = SaccadicController(
            hidden_dim=hidden_dim,
            block_size=block_size,
            temperature=gumbel_temperature,
            mask_fixated=mask_fixated,
        )

        # Global peripheral map update: cross-attention from peripheral blocks to foveal output
        self.map_update_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.map_update_norm = nn.LayerNorm(hidden_dim)
        # Confidence gate: learned per-saccade blending weight
        # Input: saccade step index (scalar) -> blending alpha
        self.map_update_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Output projection: broadcast state back to sequence length
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        peripheral_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — token embeddings
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
            peripheral_source: (batch, seq_len, hidden_dim) — if provided, used to
                build the peripheral map instead of x (e.g., layer-5 outputs)

        Returns:
            output: (batch, seq_len, hidden_dim) — contextualized representations
            info: dict with fixation_points, fixation_logits for visualization/loss
        """
        batch, seq_len, hidden_dim = x.shape  # (B, N, D)

        # Phase 1: Peripheral encoding — O(n)
        # Use peripheral_source (e.g., layer-5 outputs) if available, else x
        periph_input = peripheral_source if peripheral_source is not None else x
        # peripheral_map: (batch, num_blocks, D)
        peripheral_map = self.peripheral_encoder(periph_input, attention_mask=attention_mask)
        num_blocks = peripheral_map.shape[1]

        # Build block-level attention mask if needed
        block_mask = None
        if attention_mask is not None:
            # block_mask: (batch, num_blocks) — a block is valid if any token in it is valid
            block_mask = attention_mask.unfold(1, self.block_size, self.block_size)
            # If seq wasn't divisible by block_size, peripheral_encoder padded it,
            # so we need to handle the last partial block
            if block_mask.shape[1] < num_blocks:
                pad_blocks = num_blocks - block_mask.shape[1]
                block_mask = torch.nn.functional.pad(block_mask, (0, 0, 0, pad_blocks), value=0)
            block_mask = block_mask.any(dim=-1).float()  # (batch, num_blocks)

        # Phase 2: Iterative saccadic fixation
        # Initialize state as mean of peripheral map: (batch, D)
        if block_mask is not None:
            # Masked mean
            mask_expanded = block_mask.unsqueeze(-1)  # (batch, num_blocks, 1)
            state = (peripheral_map * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            state = peripheral_map.mean(dim=1)  # (batch, D)

        fixation_points = []
        fixation_logits_list = []
        fixation_history = []
        accumulated_context = None  # grows with each saccade

        for t in range(self.num_saccades):
            # Decide where to look: controller outputs token-level position
            # fixation_point: (batch,), logits: (batch, num_blocks), block_idx: (batch,)
            fixation_point, logits, block_idx = self.saccadic_controller(
                peripheral_map, state,
                fixation_history=fixation_history,
                attention_mask=block_mask,
            )
            fixation_points.append(fixation_point)
            fixation_logits_list.append(logits)
            fixation_history.append(block_idx)

            # Process at high resolution: foveal attention over accumulated context
            # state: (batch, D), accumulated_context: (batch, t*window_size, D)
            state, accumulated_context = self.foveal_processor(
                x, fixation_point, state, accumulated_context=accumulated_context,
            )

            # Global peripheral map update: all blocks attend to accumulated foveal context
            # This lets the controller make better-informed decisions on subsequent saccades
            normed_map = self.map_update_norm(peripheral_map)
            map_delta, _ = self.map_update_attn(
                normed_map,                    # queries: peripheral blocks
                accumulated_context,           # keys: all foveal tokens seen so far
                accumulated_context,           # values
            )
            # Confidence-weighted blending: early saccades update less than later ones
            step_input = torch.tensor([[t / self.num_saccades]], device=x.device, dtype=x.dtype)
            alpha = self.map_update_gate(step_input)  # (1, 1)
            peripheral_map = peripheral_map + alpha * map_delta

        # Phase 3: Output projection
        # Broadcast updated state back to every token position
        # state: (batch, D) -> (batch, 1, D) -> (batch, N, D)
        state_broadcast = state.unsqueeze(1).expand(-1, seq_len, -1)

        # Residual connection: combine original tokens with saccadic state
        # output: (batch, N, D)
        output = x + self.output_proj(self.output_norm(state_broadcast))

        info = {
            'fixation_points': fixation_points,       # list of (batch,) tensors
            'fixation_logits': fixation_logits_list,   # list of (batch, num_blocks) tensors
        }

        return output, info  # (batch, seq_len, hidden_dim), dict
