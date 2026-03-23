"""GPT-2 with Saccadic Attention: surgical replacement of attention in selected layers."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

from src.saccadic_attention import SaccadicAttention


class SaccadicGPT2Block(nn.Module):
    """A GPT-2 transformer block with saccadic attention replacing standard attention.

    Keeps the original FFN, LayerNorms, and residual connections.
    Only the attention mechanism is replaced.
    """

    def __init__(
        self,
        original_block: nn.Module,
        hidden_dim: int,
        num_heads: int,
        num_saccades: int = 5,
        window_size: int = 128,
        block_size: int = 32,
        gumbel_temperature: float = 1.0,
        mask_fixated: bool = False,
    ):
        super().__init__()

        # Keep original LayerNorms and FFN (frozen)
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.mlp = original_block.mlp

        # Replace attention with saccadic attention (trainable)
        self.saccadic_attn = SaccadicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_saccades=num_saccades,
            window_size=window_size,
            block_size=block_size,
            gumbel_temperature=gumbel_temperature,
            mask_fixated=mask_fixated,
        )

        # Copy pretrained Q/K/V weights into foveal processor where possible
        self._init_foveal_from_pretrained(original_block)

    def _init_foveal_from_pretrained(self, original_block: nn.Module):
        """Copy pretrained attention weights into the foveal processor's MHA."""
        pretrained_attn = original_block.attn
        foveal_attn = self.saccadic_attn.foveal_processor.attn

        # GPT-2 stores Q/K/V as a single Conv1D: c_attn.weight is (hidden_dim, 3*hidden_dim)
        # nn.MultiheadAttention stores them as in_proj_weight: (3*hidden_dim, hidden_dim)
        c_attn_weight = pretrained_attn.c_attn.weight.data  # (hidden_dim, 3*hidden_dim)
        c_attn_bias = pretrained_attn.c_attn.bias.data      # (3*hidden_dim,)

        # Conv1D in HF GPT-2 stores weights transposed: (in_features, out_features)
        # nn.MultiheadAttention in_proj_weight is (3*hidden_dim, hidden_dim)
        hidden_dim = c_attn_weight.shape[0]

        if hasattr(foveal_attn, 'in_proj_weight') and foveal_attn.in_proj_weight is not None:
            # in_proj_weight: (3*hidden_dim, hidden_dim)
            foveal_attn.in_proj_weight.data.copy_(c_attn_weight.t())
            if foveal_attn.in_proj_bias is not None:
                foveal_attn.in_proj_bias.data.copy_(c_attn_bias)

        # Copy output projection: GPT-2 c_proj -> MHA out_proj
        c_proj_weight = pretrained_attn.c_proj.weight.data  # (hidden_dim, hidden_dim)
        c_proj_bias = pretrained_attn.c_proj.bias.data      # (hidden_dim,)
        foveal_attn.out_proj.weight.data.copy_(c_proj_weight.t())
        foveal_attn.out_proj.bias.data.copy_(c_proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        peripheral_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            peripheral_source: (batch, seq_len, hidden_dim) — if provided, used to build
                the peripheral map instead of hidden_states (e.g., layer-5 outputs)

        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
            info: dict with fixation info
        """
        # Pre-norm for attention
        # residual: (batch, seq_len, D)
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        # Saccadic attention (replaces standard self-attention)
        # Use peripheral_source for building the peripheral map if available
        # attn_output: (batch, seq_len, D), info: dict
        attn_output, info = self.saccadic_attn(
            hidden_states,
            attention_mask=attention_mask,
            peripheral_source=peripheral_source,
        )
        hidden_states = residual + attn_output

        # FFN (standard GPT-2 MLP)
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, info


class GPT2Saccadic(nn.Module):
    """GPT-2 with saccadic attention in selected layers.

    All original GPT-2 parameters are frozen. Only saccadic components are trainable.
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        saccadic_layers: list[int] | None = None,
        num_saccades: int = 5,
        window_size: int = 128,
        block_size: int = 32,
        gumbel_temperature: float = 1.0,
        mask_fixated: bool = False,
    ):
        super().__init__()

        if saccadic_layers is None:
            saccadic_layers = [6, 7, 8, 9, 10, 11]
        self.saccadic_layers = saccadic_layers

        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        config = self.gpt2.config
        hidden_dim = config.n_embd
        num_heads = config.n_head

        # Replace attention in specified layers
        self.saccadic_blocks = nn.ModuleDict()
        for layer_idx in saccadic_layers:
            original_block = self.gpt2.transformer.h[layer_idx]
            saccadic_block = SaccadicGPT2Block(
                original_block=original_block,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_saccades=num_saccades,
                window_size=window_size,
                block_size=block_size,
                gumbel_temperature=gumbel_temperature,
                mask_fixated=mask_fixated,
            )
            self.saccadic_blocks[str(layer_idx)] = saccadic_block

        # Freeze all original GPT-2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Unfreeze only saccadic components (peripheral encoder, controller, output proj)
        # The foveal processor's MHA weights were copied from GPT-2 — keep them trainable
        # so they can adapt to the new windowed context
        for block in self.saccadic_blocks.values():
            for param in block.saccadic_attn.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) — for language modeling loss

        Returns:
            dict with 'loss', 'logits', 'fixation_info'
        """
        # Get token + position embeddings from GPT-2
        device = input_ids.device
        batch, seq_len = input_ids.shape

        # Embeddings
        # GPT-2's position embedding table has max 1024 entries.
        # For longer sequences, clamp position IDs — saccadic attention
        # handles long-range dependencies, not position embeddings.
        max_pos = self.gpt2.config.n_positions  # 1024 for GPT-2
        inputs_embeds = self.gpt2.transformer.wte(input_ids)       # (batch, seq_len, D)
        position_ids = torch.arange(seq_len, device=device).clamp(max=max_pos - 1).unsqueeze(0)
        position_embeds = self.gpt2.transformer.wpe(position_ids)  # (1, seq_len, D)
        hidden_states = inputs_embeds + position_embeds            # (batch, seq_len, D)
        hidden_states = self.gpt2.transformer.drop(hidden_states)

        # Cache position is required by newer transformers for causal mask generation
        cache_position = torch.arange(seq_len, device=device)

        # The last non-saccadic layer's output serves as peripheral source.
        # For default saccadic_layers=[6..11], this is layer 5's output —
        # contextualized representations that make much richer peripheral maps.
        peripheral_source = None
        first_saccadic = min(self.saccadic_layers)

        # Run through all transformer layers
        all_fixation_info = {}
        for layer_idx, block in enumerate(self.gpt2.transformer.h):
            if str(layer_idx) in self.saccadic_blocks:
                # Use saccadic block, passing peripheral_source from pre-saccadic layers
                hidden_states, info = self.saccadic_blocks[str(layer_idx)](
                    hidden_states,
                    attention_mask=attention_mask,
                    peripheral_source=peripheral_source,
                )
                all_fixation_info[layer_idx] = info
            else:
                # Use original GPT-2 block (returns tensor directly in newer transformers)
                hidden_states = block(hidden_states, cache_position=cache_position)
                # Capture output of the last non-saccadic layer as peripheral source
                if layer_idx == first_saccadic - 1:
                    peripheral_source = hidden_states.detach()  # detach to avoid backprop through frozen layers

        # Final layer norm
        hidden_states = self.gpt2.transformer.ln_f(hidden_states)  # (batch, seq_len, D)

        # LM head
        logits = self.gpt2.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            'loss': loss,
            'logits': logits,
            'fixation_info': all_fixation_info,
        }

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def set_gumbel_temperature(self, temperature: float):
        """Update Gumbel-softmax temperature across all saccadic layers."""
        for block in self.saccadic_blocks.values():
            block.saccadic_attn.saccadic_controller.temperature = temperature
