"""Saccadic Qwen: Qwen2.5-1.5B frozen + soft-routing saccadic adapter.

Architecture:
  Qwen2.5-1.5B (frozen, all layers) →
  Peripheral encoder: full 1536-dim → conv+std+max stats → project to 128-dim
  Foveal path: project 1536→128 for token-level processing
  2 saccadic layers with SOFT TOP-K ROUTING (no Gumbel-softmax)
  Project 128→1536 + residual → task head

Soft routing: during training, process top-K blocks weighted by softmax scores.
Every block's gradient flows DIRECTLY through the softmax — no discrete selection.
Temperature anneals from 5.0 (soft) to 0.1 (hard) over training.

~9M trainable params. All Qwen params frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K_RATIO = 0.1  # process 10% of blocks per saccade (min 8, max 32)

# ── Saccadic Components ───────────────────────────────────────────────────────

class PeripheralEncoder(nn.Module):
    """Split-path: operates on FULL Qwen dim for stats, projects to sacc_dim."""
    def __init__(self, base_dim, sacc_dim, block_size):
        super().__init__()
        self.block_size = block_size
        mid = 256
        self.conv_proj = nn.Linear(base_dim, mid)
        self.conv = nn.Conv1d(mid, mid, kernel_size=block_size, stride=block_size)
        self.std_proj = nn.Linear(base_dim, mid)
        self.max_proj = nn.Linear(base_dim, mid)
        self.project = nn.Linear(mid * 3, sacc_dim)
        self.norm = nn.LayerNorm(sacc_dim)
        self.pos_emb = nn.Embedding(16384, sacc_dim)

    def forward(self, x):
        B, N, D = x.shape
        pad = (self.block_size - N % self.block_size) % self.block_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        nb = x.shape[1] // self.block_size
        x_mid = self.conv_proj(x)
        conv_out = self.conv(x_mid.transpose(1, 2)).transpose(1, 2)
        blocks = x.reshape(B, nb, self.block_size, D)
        std_out = self.std_proj(blocks.std(dim=2))
        max_out = self.max_proj(blocks.max(dim=2).values)
        combined = torch.cat([conv_out, std_out, max_out], dim=-1)
        out = self.norm(self.project(combined))
        return out + self.pos_emb(torch.arange(nb, device=x.device))


class SaccadicController(nn.Module):
    """Soft top-K routing. Fully differentiable — no Gumbel-softmax."""
    def __init__(self, dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.temperature = 5.0

    def forward(self, peripheral_map, state):
        scores = torch.einsum(
            'bd,bmd->bm', self.q_proj(state), self.k_proj(peripheral_map)
        ) / math.sqrt(self.dim)
        logits = scores

        if self.training:
            n_blocks = scores.shape[1]
            top_k = max(8, min(32, int(n_blocks * TOP_K_RATIO)))
            topk_vals, topk_idx = torch.topk(
                scores, k=min(top_k, n_blocks), dim=-1)
            topk_weights = F.softmax(topk_vals / self.temperature, dim=-1)
            best_fp = (topk_idx[:, 0] * self.block_size).long()
        else:
            topk_idx = scores.argmax(dim=-1, keepdim=True)
            topk_weights = torch.ones(scores.shape[0], 1, device=scores.device)
            best_fp = (topk_idx[:, 0] * self.block_size).long()

        return best_fp, logits, topk_idx, topk_weights


class FovealProcessor(nn.Module):
    def __init__(self, dim, n_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def extract_window(self, x, center):
        B, N, D = x.shape
        h = self.window_size // 2
        windows = []
        for i in range(B):
            c = center[i].item()
            s, e = max(0, c - h), min(N, c + h)
            if e - s < self.window_size:
                s = max(0, e - self.window_size) if s > 0 else 0
                e = min(N, s + self.window_size)
            w = x[i, s:e]
            if w.shape[0] < self.window_size:
                w = F.pad(w, (0, 0, 0, self.window_size - w.shape[0]))
            windows.append(w)
        return torch.stack(windows)

    def forward_single(self, x, center, state, accumulated_context=None):
        win = self.extract_window(x, center)
        ctx = torch.cat(
            [state.unsqueeze(1)] +
            ([accumulated_context, win] if accumulated_context is not None else [win]),
            dim=1,
        )
        out, _ = self.attn(self.norm1(ctx), self.norm1(ctx), self.norm1(ctx))
        ctx = ctx + out
        cls_out = ctx[:, 0]
        cls_out = cls_out + self.ffn(self.norm2(cls_out))
        return cls_out, win


class SaccadicLayer(nn.Module):
    """Split-path saccadic layer with soft top-K routing."""
    def __init__(self, base_dim, sacc_dim, n_heads, block_size, window_size, num_saccades):
        super().__init__()
        self.num_saccades = num_saccades
        self.peripheral = PeripheralEncoder(base_dim, sacc_dim, block_size)
        self.controller = SaccadicController(sacc_dim, block_size)
        self.foveal = FovealProcessor(sacc_dim, n_heads, window_size)
        self.map_attn = nn.MultiheadAttention(sacc_dim, n_heads, batch_first=True)
        self.map_norm = nn.LayerNorm(sacc_dim)
        self.map_gate = nn.Sequential(
            nn.Linear(1, sacc_dim), nn.GELU(), nn.Linear(sacc_dim, 1), nn.Sigmoid())
        self.out_proj = nn.Linear(sacc_dim, sacc_dim)
        self.out_norm = nn.LayerNorm(sacc_dim)
        self.ln1 = nn.LayerNorm(sacc_dim)
        self.ln2 = nn.LayerNorm(sacc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sacc_dim, sacc_dim * 4), nn.GELU(), nn.Linear(sacc_dim * 4, sacc_dim))

    def forward(self, x_sacc, x_full):
        B, N, D = x_sacc.shape
        residual = x_sacc
        h = self.ln1(x_sacc)
        pmap = self.peripheral(x_full)
        state = pmap.mean(dim=1)
        fixation_points = []
        fixation_logits = []
        acc_ctx = None

        for t in range(self.num_saccades):
            fp, logits, topk_idx, topk_w = self.controller(pmap, state)
            fixation_points.append(fp)
            fixation_logits.append(logits)

            if self.training:
                K = topk_idx.shape[1]
                weighted_state = torch.zeros_like(state)
                best_win = None
                for k in range(K):
                    centers = (topk_idx[:, k] * self.controller.block_size).long()
                    s_k, win_k = self.foveal.forward_single(h, centers, state, acc_ctx)
                    weighted_state = weighted_state + topk_w[:, k].unsqueeze(-1) * s_k
                    if k == 0:
                        best_win = win_k
                state = weighted_state
                acc_ctx = torch.cat([acc_ctx, best_win], 1) if acc_ctx is not None else best_win
            else:
                s_new, win_new = self.foveal.forward_single(h, fp, state, acc_ctx)
                state = s_new
                acc_ctx = torch.cat([acc_ctx, win_new], 1) if acc_ctx is not None else win_new

            delta, _ = self.map_attn(self.map_norm(pmap), acc_ctx, acc_ctx)
            alpha = self.map_gate(
                torch.tensor([[t / self.num_saccades]], device=x_sacc.device, dtype=x_sacc.dtype))
            pmap = pmap + alpha * delta

        out = self.out_proj(self.out_norm(state.unsqueeze(1).expand(-1, N, -1)))
        x_sacc = residual + out
        x_sacc = x_sacc + self.mlp(self.ln2(x_sacc))
        return x_sacc, {'fixation_points': fixation_points, 'fixation_logits': fixation_logits}


# ── Main Model ────────────────────────────────────────────────────────────────

class SaccadicQwen(nn.Module):
    """Qwen2.5-1.5B (frozen) + soft-routing saccadic adapter."""

    def __init__(
        self,
        model_name='Qwen/Qwen2.5-1.5B',
        saccadic_dim=128,
        block_size=8,
        window_size=64,
        num_saccades=3,
        n_heads=4,
        task_head=None,
    ):
        super().__init__()
        self.num_saccades = num_saccades

        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, trust_remote_code=True)
        self.base_dim = self.qwen.config.hidden_size
        for p in self.qwen.parameters():
            p.requires_grad = False

        self.proj_down = nn.Linear(self.base_dim, saccadic_dim)
        self.proj_up = nn.Linear(saccadic_dim, self.base_dim)

        self.sacc1 = SaccadicLayer(
            self.base_dim, saccadic_dim, n_heads, block_size, window_size, num_saccades)
        self.sacc2 = SaccadicLayer(
            self.base_dim, saccadic_dim, n_heads, block_size, window_size, num_saccades)

        self.ln_out = nn.LayerNorm(self.base_dim)
        self.task_head = task_head
        self._trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        with torch.no_grad():
            outputs = self.qwen(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()

        h = self.proj_down(hidden)
        h, info1 = self.sacc1(h, hidden)
        h, info2 = self.sacc2(h, hidden)
        h_up = self.proj_up(h)
        out = self.ln_out(hidden + h_up)
        last_hidden = out[:, -1, :]

        result = {'fixation_info': {0: info1, 1: info2}}
        if self.task_head is not None:
            head_out = self.task_head(last_hidden, labels)
            result.update(head_out)
        return result

    def set_gumbel_temperature(self, temp):
        self.sacc1.controller.temperature = temp
        self.sacc2.controller.temperature = temp

    def trainable_params(self):
        return self._trainable


# ── Task Heads ────────────────────────────────────────────────────────────────

class DigitClassificationHead(nn.Module):
    def __init__(self, input_dim, n_digits=7):
        super().__init__()
        self.n_digits = n_digits
        self.heads = nn.ModuleList([nn.Linear(input_dim, 10) for _ in range(n_digits)])

    def forward(self, hidden, labels=None):
        logits = [h(hidden) for h in self.heads]
        loss = None
        if labels is not None:
            loss = sum(F.cross_entropy(logits[i], labels[:, i]) for i in range(self.n_digits)) / self.n_digits
        return {'loss': loss, 'digit_logits': logits}


class MultiValueHead(nn.Module):
    def __init__(self, input_dim, n_values=4, n_digits=7):
        super().__init__()
        self.n_values = n_values
        self.n_digits = n_digits
        self.heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(input_dim, 10) for _ in range(n_digits)])
            for _ in range(n_values)
        ])

    def forward(self, hidden, labels=None):
        all_logits = []
        loss = torch.tensor(0.0, device=hidden.device) if labels is not None else None
        for v in range(self.n_values):
            v_logits = [h(hidden) for h in self.heads[v]]
            all_logits.append(v_logits)
            if labels is not None:
                for d in range(self.n_digits):
                    loss = loss + F.cross_entropy(v_logits[d], labels[:, v, d])
        if loss is not None:
            loss = loss / (self.n_values * self.n_digits)
        return {'loss': loss, 'all_logits': all_logits}


class VariableTrackingHead(nn.Module):
    def __init__(self, input_dim, n_digits=4):
        super().__init__()
        self.n_digits = n_digits
        self.heads = nn.ModuleList([nn.Linear(input_dim, 10) for _ in range(n_digits)])

    def forward(self, hidden, labels=None):
        logits = [h(hidden) for h in self.heads]
        loss = None
        if labels is not None:
            loss = sum(F.cross_entropy(logits[i], labels[:, i]) for i in range(self.n_digits)) / self.n_digits
        return {'loss': loss, 'digit_logits': logits}
