"""Saccadic Qwen: Qwen2.5-1.5B frozen + bottlenecked saccadic adapter on top.

Architecture:
  Qwen2.5-1.5B (frozen, all 28 layers) → project 1536→128 →
  2 saccadic layers (128-dim, accumulated context, peripheral map update) →
  project 128→1536 → task head

~3-5M trainable params. All Qwen params frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Saccadic Components (128-dim) ─────────────────────────────────────────────

GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_END = 0.1


class PeripheralEncoder(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.block_size = block_size
        # Learned convolutional downsampling (better than uniform average)
        self.conv = nn.Conv1d(dim, dim, kernel_size=block_size, stride=block_size)
        # Project concat(conv, std, max) → dim
        self.project = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.pos_emb = nn.Embedding(16384, dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        pad = (self.block_size - N % self.block_size) % self.block_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        n_blocks = x.shape[1] // self.block_size

        # Learned conv downsampling: (B, N, D) → (B, D, N) → conv → (B, D, nb) → (B, nb, D)
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, nb, D)

        # Statistical features per block
        blocks = x.reshape(B, n_blocks, self.block_size, D)  # (B, nb, bs, D)
        std_pool = blocks.std(dim=2)    # (B, nb, D) — high variance = unusual content
        max_pool = blocks.max(dim=2).values  # (B, nb, D) — outlier detection

        # Combine all three signals
        combined = torch.cat([conv_out, std_pool, max_pool], dim=-1)  # (B, nb, 3*D)
        out = self.norm(self.project(combined))  # (B, nb, D)
        return out + self.pos_emb(torch.arange(n_blocks, device=x.device))


class SaccadicController(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.temperature = GUMBEL_TEMP_START

    def forward(self, peripheral_map, state):
        # state: (B, D), peripheral_map: (B, M, D)
        scores = torch.einsum(
            'bd,bmd->bm',
            self.q_proj(state),
            self.k_proj(peripheral_map),
        ) / math.sqrt(self.dim)  # (B, M)
        logits = scores
        if self.training:
            sel = F.gumbel_softmax(scores, tau=self.temperature, hard=True)
            idx_f = torch.einsum(
                'bm,m->b', sel,
                torch.arange(peripheral_map.shape[1], device=scores.device, dtype=torch.float),
            )
        else:
            idx_f = scores.argmax(dim=-1).float()
        fixation_point = (idx_f * self.block_size).long()
        return fixation_point, logits, idx_f.long()


class FovealProcessor(nn.Module):
    def __init__(self, dim, n_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def extract_windows(self, x, fixation_point):
        B, N, D = x.shape
        half = self.window_size // 2
        windows = []
        for i in range(B):
            c = fixation_point[i].item()
            s = max(0, c - half)
            e = min(N, c + half)
            if e - s < self.window_size:
                s = max(0, e - self.window_size) if s > 0 else 0
                e = min(N, s + self.window_size)
            w = x[i, s:e]
            if w.shape[0] < self.window_size:
                w = F.pad(w, (0, 0, 0, self.window_size - w.shape[0]))
            windows.append(w)
        return torch.stack(windows)  # (B, ws, D)

    def forward(self, x, fixation_point, state, accumulated_context=None):
        window = self.extract_windows(x, fixation_point)  # (B, ws, D)
        cls = state.unsqueeze(1)  # (B, 1, D)
        if accumulated_context is not None:
            ctx = torch.cat([cls, accumulated_context, window], dim=1)
        else:
            ctx = torch.cat([cls, window], dim=1)
        normed = self.norm1(ctx)
        out, _ = self.attn(normed, normed, normed)
        ctx = ctx + out
        cls_out = ctx[:, 0]
        cls_out = cls_out + self.ffn(self.norm2(cls_out))
        new_acc = torch.cat([accumulated_context, window], dim=1) if accumulated_context is not None else window
        return cls_out, new_acc


class SaccadicLayer(nn.Module):
    """One full saccadic layer: peripheral encode → saccadic loop → output."""
    def __init__(self, dim, n_heads, block_size, window_size, num_saccades):
        super().__init__()
        self.num_saccades = num_saccades
        self.peripheral = PeripheralEncoder(dim, block_size)
        self.controller = SaccadicController(dim, block_size)
        self.foveal = FovealProcessor(dim, n_heads, window_size)
        # Peripheral map update
        self.map_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.map_norm = nn.LayerNorm(dim)
        self.map_gate = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        # Output
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        h = self.ln1(x)
        pmap = self.peripheral(h)
        state = pmap.mean(dim=1)
        fixation_points = []
        fixation_logits = []
        acc_ctx = None

        for t in range(self.num_saccades):
            fp, logits, _ = self.controller(pmap, state)
            fixation_points.append(fp)
            fixation_logits.append(logits)
            state, acc_ctx = self.foveal(h, fp, state, acc_ctx)
            # Global peripheral map update
            delta, _ = self.map_attn(self.map_norm(pmap), acc_ctx, acc_ctx)
            alpha = self.map_gate(
                torch.tensor([[t / self.num_saccades]], device=x.device, dtype=x.dtype))
            pmap = pmap + alpha * delta

        out = self.out_proj(self.out_norm(state.unsqueeze(1).expand(-1, N, -1)))
        x = residual + out
        x = x + self.mlp(self.ln2(x))
        info = {'fixation_points': fixation_points, 'fixation_logits': fixation_logits}
        return x, info


# ── Main Model ────────────────────────────────────────────────────────────────

class SaccadicQwen(nn.Module):
    """Qwen2.5-1.5B (frozen) + bottlenecked saccadic adapter."""

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

        # Load and freeze Qwen
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, trust_remote_code=True)
        self.base_dim = self.qwen.config.hidden_size  # 1536
        for p in self.qwen.parameters():
            p.requires_grad = False

        # Bottleneck projections (float32 for training stability)
        self.proj_down = nn.Linear(self.base_dim, saccadic_dim)
        self.proj_up = nn.Linear(saccadic_dim, self.base_dim)

        # Two saccadic layers
        self.sacc1 = SaccadicLayer(saccadic_dim, n_heads, block_size, window_size, num_saccades)
        self.sacc2 = SaccadicLayer(saccadic_dim, n_heads, block_size, window_size, num_saccades)

        # Output norm
        self.ln_out = nn.LayerNorm(self.base_dim)

        # Task head (set externally)
        self.task_head = task_head

        self._trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None, target_positions=None):
        """
        Args:
            input_ids: (B, N) token IDs
            labels: task-specific labels
            target_positions: list of target positions for supervised warmup
        Returns:
            dict with loss, predictions, fixation_info
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Run frozen Qwen
        with torch.no_grad():
            outputs = self.qwen(input_ids, output_hidden_states=True)
            # Use last hidden state, cast to float32
            hidden = outputs.hidden_states[-1].float()  # (B, N, 1536)

        # Project down to saccadic dim
        h = self.proj_down(hidden)  # (B, N, 128)

        # Two saccadic layers
        h, info1 = self.sacc1(h)
        h, info2 = self.sacc2(h)

        # Project back up + residual with Qwen output
        h_up = self.proj_up(h)  # (B, N, 1536)
        out = self.ln_out(hidden + h_up)  # (B, N, 1536)

        # Use the last token for classification
        last_hidden = out[:, -1, :]  # (B, 1536)

        # Task head
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
    """Predict N digits independently (0-9 each). For passkey/needle tasks."""
    def __init__(self, input_dim, n_digits=7):
        super().__init__()
        self.n_digits = n_digits
        self.heads = nn.ModuleList([nn.Linear(input_dim, 10) for _ in range(n_digits)])

    def forward(self, hidden, labels=None):
        logits = [h(hidden) for h in self.heads]  # list of (B, 10)
        loss = None
        if labels is not None:
            loss = sum(F.cross_entropy(logits[i], labels[:, i]) for i in range(self.n_digits)) / self.n_digits
        preds = ''.join  # will be computed in eval
        return {'loss': loss, 'digit_logits': logits}


class MultiValueHead(nn.Module):
    """Predict multiple values. For MV-NIAH (predict which values are present)."""
    def __init__(self, input_dim, n_values=4, n_digits=7):
        super().__init__()
        self.n_values = n_values
        self.n_digits = n_digits
        # Predict n_values × n_digits independently
        self.heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(input_dim, 10) for _ in range(n_digits)])
            for _ in range(n_values)
        ])

    def forward(self, hidden, labels=None):
        # labels: (B, n_values, n_digits)
        all_logits = []
        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, device=hidden.device)
        for v in range(self.n_values):
            v_logits = [h(hidden) for h in self.heads[v]]
            all_logits.append(v_logits)
            if labels is not None:
                for d in range(self.n_digits):
                    loss = loss + F.cross_entropy(v_logits[d], labels[:, v, d])
        if loss is not None:
            loss = loss / (self.n_values * self.n_digits)
        return {'loss': loss, 'all_logits': all_logits}


class WordPredictionHead(nn.Module):
    """Predict top-K words from a fixed vocabulary. For CWE/FWE tasks."""
    def __init__(self, input_dim, vocab_size=1000, k=10):
        super().__init__()
        self.k = k
        self.classifier = nn.Linear(input_dim, vocab_size)

    def forward(self, hidden, labels=None):
        # labels: (B, vocab_size) multi-hot
        logits = self.classifier(hidden)  # (B, vocab_size)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return {'loss': loss, 'word_logits': logits}


class VariableTrackingHead(nn.Module):
    """Predict final value in a variable chain. Same as digit classification."""
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
