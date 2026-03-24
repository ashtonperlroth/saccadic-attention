"""Debug: validate saccadic module works on Qwen at all.

Exact same passkey task that got 100% on tiny model:
- 5-digit passkey, char-level classification head (5 × 10-class)
- Context 2048, block_size=8, window=64, 3 saccades
- Supervised warmup weight=2.0, steps=300
- Train for 20 minutes (fixed budget), then evaluate

Also runs diagnostics: gradient flow, param counts, frozen verification.
"""

import math, random, string, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = 'Qwen/Qwen2.5-1.5B'
SACC_DIM = 128
BLOCK_SIZE = 8
WINDOW_SIZE = 64
NUM_SACCADES = 3
N_HEADS = 4
CONTEXT_LENGTH = 2048
LR = 1e-3
BATCH_SIZE = 4
SUPERVISED_WARMUP_STEPS = 300
SUPERVISED_WARMUP_WEIGHT = 2.0
GUMBEL_ANNEAL_STEPS = 800
PATIENCE = 2000
VAL_EVERY = 50
MAX_TIME = 1200  # 20 minutes
NUM_TRAIN = 3000
NUM_VAL = 100
NUM_EVAL = 200

def log(msg): print(msg, file=sys.stderr, flush=True)


# ── Simple Passkey Dataset (uses Qwen tokenizer) ─────────────────────────────

FILLER = [
    "The weather was pleasant and the sky was clear.",
    "Several researchers gathered to discuss the latest findings.",
    "The library contained thousands of books on various topics.",
    "Traffic moved slowly through the busy intersection.",
    "A gentle breeze rustled through the autumn leaves.",
    "The project deadline was approaching rapidly.",
    "Students worked diligently on their assignments.",
    "The old building stood at the corner of the street.",
    "New developments in technology continued to emerge.",
    "The garden was well maintained throughout the year.",
]


class PasskeyDataset(Dataset):
    def __init__(self, n, ctx_len, tokenizer, seed=42):
        self.rng = random.Random(seed)
        self.filler_enc = [tokenizer.encode(" " + s) for s in FILLER]
        self.prompt = tokenizer.encode(" What is the secret number? The secret number is")
        self.samples = [self._make(i, ctx_len, tokenizer) for i in range(n)]

    def _make(self, idx, ctx_len, tokenizer):
        rng = random.Random(self.rng.randint(0, 2**32) + idx)
        pk = ''.join(rng.choices(string.digits, k=5))
        pk_ids = tokenizer.encode(f" The secret number is {pk}.")
        ans_ids = tokenizer.encode(" " + pk)
        budget = ctx_len - len(pk_ids) - len(self.prompt) - len(ans_ids)
        fill = []
        while len(fill) < budget:
            fill.extend(rng.choice(self.filler_enc))
        fill = fill[:budget]
        pos = rng.randint(0, len(fill))
        full = fill[:pos] + pk_ids + fill[pos:] + self.prompt + ans_ids
        if len(full) > ctx_len: full = full[:ctx_len]
        elif len(full) < ctx_len: full += [tokenizer.eos_token_id] * (ctx_len - len(full))
        return (torch.tensor(full[:ctx_len], dtype=torch.long),
                torch.tensor([int(d) for d in pk], dtype=torch.long), pk, pos)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids, dl, pk, pos = self.samples[i]
        return {'input_ids': ids, 'digit_labels': dl, 'passkey': pk, 'passkey_position': pos}

def collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'digit_labels': torch.stack([b['digit_labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
    }


# ── Saccadic Components (identical to tiny model) ────────────────────────────

class PeripheralEncoder(nn.Module):
    """Operates on FULL Qwen dim (1536) for statistics, uses lightweight conv.

    Conv projects 1536→256 (cheap), then combines with std/max stats.
    Each stat is projected independently 1536→256, then concat → 128.
    Total: ~1.5M params instead of ~18M.
    """
    def __init__(self, base_dim, sacc_dim, bs):
        super().__init__()
        self.bs = bs
        mid = 256  # intermediate projection dim
        # Lightweight conv: project down THEN convolve (much cheaper)
        self.conv_proj = nn.Linear(base_dim, mid)
        self.conv = nn.Conv1d(mid, mid, kernel_size=bs, stride=bs)
        # Independent projections for std and max (1536→256 each)
        self.std_proj = nn.Linear(base_dim, mid)
        self.max_proj = nn.Linear(base_dim, mid)
        # Combine 3×256 → sacc_dim
        self.project = nn.Linear(mid * 3, sacc_dim)
        self.norm = nn.LayerNorm(sacc_dim)
        self.pos_emb = nn.Embedding(16384, sacc_dim)

    def forward(self, x):
        # x: (B, N, base_dim=1536) — full Qwen hidden states
        B, N, D = x.shape
        pad = (self.bs - N % self.bs) % self.bs
        if pad: x = F.pad(x, (0, 0, 0, pad))
        nb = x.shape[1] // self.bs

        # Conv path: project 1536→256 then convolve
        x_mid = self.conv_proj(x)  # (B, N, 256)
        conv_out = self.conv(x_mid.transpose(1, 2)).transpose(1, 2)  # (B, nb, 256)

        # Stats at full dim, then project
        blocks = x.reshape(B, nb, self.bs, D)  # (B, nb, bs, 1536)
        std_full = blocks.std(dim=2)       # (B, nb, 1536)
        max_full = blocks.max(dim=2).values  # (B, nb, 1536)
        std_out = self.std_proj(std_full)  # (B, nb, 256)
        max_out = self.max_proj(max_full)  # (B, nb, 256)

        combined = torch.cat([conv_out, std_out, max_out], dim=-1)  # (B, nb, 768)
        out = self.norm(self.project(combined))  # (B, nb, 128)
        return out + self.pos_emb(torch.arange(nb, device=x.device))

TOP_K = 16  # number of blocks to process per saccade

class Controller(nn.Module):
    def __init__(self, dim, bs):
        super().__init__()
        self.bs, self.dim = bs, dim
        self.qp = nn.Linear(dim, dim); self.kp = nn.Linear(dim, dim)
        self.temperature = 5.0  # start soft, anneal to 0.1

    def forward(self, pm, state):
        # pm: (B, M, D), state: (B, D)
        s = torch.einsum('bd,bmd->bm', self.qp(state), self.kp(pm)) / math.sqrt(self.dim)
        logits = s  # (B, M) — raw scores for supervised warmup loss

        if self.training:
            # Soft routing: weights over ALL blocks (fully differentiable)
            weights = F.softmax(s / self.temperature, dim=-1)  # (B, M)
            # Top-K for efficiency
            topk_vals, topk_idx = torch.topk(s, k=min(TOP_K, s.shape[1]), dim=-1)
            topk_weights = F.softmax(topk_vals / self.temperature, dim=-1)  # (B, K)
            # Best fixation point for logging
            best_fp = (topk_idx[:, 0] * self.bs).long()
        else:
            # Hard selection at inference
            topk_idx = s.argmax(dim=-1, keepdim=True)  # (B, 1)
            topk_weights = torch.ones(s.shape[0], 1, device=s.device)
            best_fp = (topk_idx[:, 0] * self.bs).long()

        return best_fp, logits, topk_idx, topk_weights


class Foveal(nn.Module):
    def __init__(self, dim, nh, ws):
        super().__init__()
        self.ws = ws
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def _ext_single(self, x, center):
        """Extract one window centered at `center` for all batch elements."""
        B, N, D = x.shape; h = self.ws // 2; ws = []
        for i in range(B):
            c = center[i].item(); s, e = max(0, c-h), min(N, c+h)
            if e-s < self.ws: s = max(0, e-self.ws) if s>0 else 0; e = min(N, s+self.ws)
            w = x[i, s:e]
            if w.shape[0] < self.ws: w = F.pad(w, (0, 0, 0, self.ws-w.shape[0]))
            ws.append(w)
        return torch.stack(ws)  # (B, ws, D)

    def forward_single(self, x, center, state, acc=None):
        """Process one window. Returns (state_update, window)."""
        win = self._ext_single(x, center)  # (B, ws, D)
        ctx = torch.cat([state.unsqueeze(1)] + ([acc, win] if acc is not None else [win]), 1)
        out, _ = self.attn(self.n1(ctx), self.n1(ctx), self.n1(ctx))
        ctx = ctx + out
        cl = ctx[:, 0]
        cl = cl + self.ff(self.n2(cl))
        return cl, win  # state update and raw window

class SaccadicLayer(nn.Module):
    """Split-path saccadic layer.

    Peripheral encoder runs on full-dim hidden states (1536).
    Foveal processor runs on projected tokens (128-dim).
    """
    def __init__(self, base_dim, sacc_dim, nh, bs, ws, ns):
        super().__init__()
        self.ns = ns
        # Peripheral encoder: full dim → sacc_dim
        self.pe = PeripheralEncoder(base_dim, sacc_dim, bs)
        # Controller and foveal in sacc_dim space
        self.ctrl = Controller(sacc_dim, bs)
        self.fov = Foveal(sacc_dim, nh, ws)
        # Peripheral map update in sacc_dim space
        self.ma = nn.MultiheadAttention(sacc_dim, nh, batch_first=True)
        self.mn = nn.LayerNorm(sacc_dim)
        self.mg = nn.Sequential(nn.Linear(1, sacc_dim), nn.GELU(), nn.Linear(sacc_dim, 1), nn.Sigmoid())
        # Output projection
        self.op = nn.Linear(sacc_dim, sacc_dim); self.on = nn.LayerNorm(sacc_dim)
        self.ln1 = nn.LayerNorm(sacc_dim); self.ln2 = nn.LayerNorm(sacc_dim)
        self.mlp = nn.Sequential(nn.Linear(sacc_dim, sacc_dim*4), nn.GELU(), nn.Linear(sacc_dim*4, sacc_dim))

    def forward(self, x_sacc, x_full):
        """
        x_sacc: (B, N, sacc_dim=128) — projected tokens for foveal processing
        x_full: (B, N, base_dim=1536) — full Qwen hidden states for peripheral encoding
        """
        B, N, D = x_sacc.shape
        res = x_sacc
        h = self.ln1(x_sacc)
        # Peripheral map from FULL-dim Qwen hidden states
        pm = self.pe(x_full)  # (B, nb, 128) — rich features projected to sacc_dim
        st = pm.mean(1)
        fps, fls = [], []; ac = None
        for t in range(self.ns):
            fp, lg, topk_idx, topk_w = self.ctrl(pm, st)
            fps.append(fp); fls.append(lg)

            if self.training:
                # Soft routing: process top-K blocks, weight outputs
                K = topk_idx.shape[1]
                weighted_state = torch.zeros_like(st)
                best_win = None
                for k in range(K):
                    centers = (topk_idx[:, k] * self.ctrl.bs).long()
                    s_k, win_k = self.fov.forward_single(h, centers, st, ac)
                    weighted_state = weighted_state + topk_w[:, k].unsqueeze(-1) * s_k
                    if k == 0:
                        best_win = win_k  # accumulate the highest-scored window
                st = weighted_state
                ac = torch.cat([ac, best_win], 1) if ac is not None else best_win
            else:
                # Hard selection at inference — single window
                s_new, win_new = self.fov.forward_single(h, fp, st, ac)
                st = s_new
                ac = torch.cat([ac, win_new], 1) if ac is not None else win_new

            d, _ = self.ma(self.mn(pm), ac, ac)
            a = self.mg(torch.tensor([[t/self.ns]], device=x_sacc.device, dtype=x_sacc.dtype))
            pm = pm + a * d
        x_sacc = res + self.op(self.on(st.unsqueeze(1).expand(-1, N, -1)))
        x_sacc = x_sacc + self.mlp(self.ln2(x_sacc))
        return x_sacc, {'fixation_points': fps, 'fixation_logits': fls}


# ── Model ─────────────────────────────────────────────────────────────────────

class QwenSaccadicPasskey(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, trust_remote_code=True)
        self.base_dim = self.qwen.config.hidden_size  # 1536
        self.max_pos = getattr(self.qwen.config, 'max_position_embeddings', 32768)

        # Freeze ALL Qwen
        for p in self.qwen.parameters(): p.requires_grad = False

        # Foveal path projection (1536 → 128 for foveal processing)
        self.proj_down = nn.Linear(self.base_dim, SACC_DIM)
        self.proj_up = nn.Linear(SACC_DIM, self.base_dim)

        # 2 saccadic layers (peripheral gets full 1536, foveal gets 128)
        self.sacc1 = SaccadicLayer(self.base_dim, SACC_DIM, N_HEADS, BLOCK_SIZE, WINDOW_SIZE, NUM_SACCADES)
        self.sacc2 = SaccadicLayer(self.base_dim, SACC_DIM, N_HEADS, BLOCK_SIZE, WINDOW_SIZE, NUM_SACCADES)

        # Classification head
        self.ln_out = nn.LayerNorm(self.base_dim)
        self.digit_heads = nn.ModuleList([nn.Linear(self.base_dim, 10) for _ in range(5)])

        self._tp = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        with torch.no_grad():
            out = self.qwen(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()  # (B, N, 1536)

        h = self.proj_down(hidden)  # (B, N, 128) — for foveal path
        # Peripheral path gets full 1536-dim, foveal path gets 128-dim
        h, info1 = self.sacc1(h, hidden)
        h, info2 = self.sacc2(h, hidden)
        h_up = self.proj_up(h)  # (B, N, 1536)
        out_h = self.ln_out(hidden + h_up)

        last = out_h[:, -1, :]  # (B, 1536)
        dl = [head(last) for head in self.digit_heads]
        loss = sum(F.cross_entropy(dl[i], labels[:, i]) for i in range(5)) / 5 if labels is not None else None
        return {'loss': loss, 'digit_logits': dl, 'fixation_info': {0: info1, 1: info2}}

    def set_gumbel_temperature(self, t):
        self.sacc1.ctrl.temperature = t
        self.sacc2.ctrl.temperature = t


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_diagnostics(model, device, tokenizer):
    log('\n=== DIAGNOSTICS ===')

    # 1. Param counts
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = model._tp
    log(f'Frozen params: {frozen:,}')
    log(f'Trainable params: {trainable:,}')

    # 2. Verify Qwen is truly frozen
    qwen_grads = any(p.requires_grad for p in model.qwen.parameters())
    log(f'Qwen has requires_grad=True params: {qwen_grads} (should be False)')

    # 3. Forward pass + gradient check
    x = torch.randint(0, 1000, (1, 256)).to(device)
    labels = torch.randint(0, 10, (1, 5)).to(device)
    out = model(x, labels=labels)
    log(f'Loss: {out["loss"].item():.4f} (random init, should be ~log(10)≈2.3)')
    out['loss'].backward()

    # Check gradient flow
    has_grad = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            has_grad[name] = p.grad is not None and p.grad.abs().sum() > 0
    grad_ok = sum(has_grad.values())
    grad_total = len(has_grad)
    log(f'Params with nonzero gradients: {grad_ok}/{grad_total}')
    if grad_ok < grad_total:
        for name, ok in has_grad.items():
            if not ok:
                log(f'  NO GRADIENT: {name}')

    # 4. Check projection layers
    model.zero_grad()
    out = model(x, labels=labels)
    out['loss'].backward()
    pd_grad = model.proj_down.weight.grad.abs().mean().item()
    pu_grad = model.proj_up.weight.grad.abs().mean().item()
    log(f'proj_down grad magnitude: {pd_grad:.6f}')
    log(f'proj_up grad magnitude: {pu_grad:.6f}')

    # 5. Check hidden state range
    with torch.no_grad():
        qout = model.qwen(x, output_hidden_states=True)
        h = qout.hidden_states[-1].float()
        log(f'Qwen hidden state: mean={h.mean():.4f}, std={h.std():.4f}, '
            f'min={h.min():.4f}, max={h.max():.4f}')
        h_proj = model.proj_down(h)
        log(f'After proj_down: mean={h_proj.mean():.4f}, std={h_proj.std():.4f}')

    # 6. Check fixation points
    model.eval()
    with torch.no_grad():
        out = model(x)
        for layer_idx, info in out['fixation_info'].items():
            fps = [fp[0].item() for fp in info['fixation_points']]
            log(f'Layer {layer_idx} fixation points: {fps}')

    log('=== DIAGNOSTICS COMPLETE ===\n')


# ── Training ──────────────────────────────────────────────────────────────────

def routing_temp(step):
    """Anneal from 5.0 (soft) to 0.1 (hard) over training."""
    p = min(step / max(GUMBEL_ANNEAL_STEPS, 1), 1.0)
    return 5.0 + (0.1 - 5.0) * p

def quick_val(model, val_loader, device):
    model.eval(); cor = tot = 0
    with torch.no_grad():
        for b in val_loader:
            ids = b['input_ids'].to(device); out = model(ids)
            for i in range(ids.shape[0]):
                pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                if pred == b['passkey'][i]: cor += 1
                tot += 1
    model.train(); return cor / max(tot, 1)

def train_and_eval(model, tokenizer, device):
    train_ds = PasskeyDataset(NUM_TRAIN, CONTEXT_LENGTH, tokenizer, seed=42)
    val_ds = PasskeyDataset(NUM_VAL, CONTEXT_LENGTH, tokenizer, seed=77777)
    test_ds = PasskeyDataset(NUM_EVAL, CONTEXT_LENGTH, tokenizer, seed=99999)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    test_ld = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR, weight_decay=0.01)
    def lrf(s):
        if s < 50: return s / 50
        return 0.5 * (1 + math.cos(math.pi * (s - 50) / 5000))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrf)

    model.train()
    t0 = time.time(); step = 0; best_val = 0.0; since = 0

    while True:
        for b in train_ld:
            elapsed = time.time() - t0
            if elapsed >= MAX_TIME:
                log(f'MAX TIME: {step} steps, {elapsed:.0f}s, best_val={best_val:.4f}')
                break
            if step > 0 and step % VAL_EVERY == 0:
                va = quick_val(model, val_ld, device)
                if va > best_val:
                    best_val = va; since = 0
                    log(f'  step {step} | val={va:.4f} (NEW BEST) | {elapsed:.0f}s')
                    if va >= 1.0:
                        log(f'  PERFECT at step {step}'); break
                else:
                    since += VAL_EVERY
                    if step % 200 == 0:
                        log(f'  step {step} | val={va:.4f} (best={best_val:.4f}, plat={since}) | {elapsed:.0f}s')
                if since >= PATIENCE:
                    log(f'  PLATEAU at step {step}, best={best_val:.4f}'); break

            ids = b['input_ids'].to(device); dl = b['digit_labels'].to(device)
            model.set_gumbel_temperature(routing_temp(step))
            out = model(ids, labels=dl); loss = out['loss']

            # Entropy bonus
            te = torch.tensor(0.0); ec = 0
            for _, inf in out['fixation_info'].items():
                for lg in inf['fixation_logits']:
                    p = F.softmax(lg, -1)
                    te = te + (-(p*(p+1e-8).log()).sum(-1).mean()).to(te.device); ec += 1
            if ec: loss = loss - 0.01 * te / ec

            # Supervised warmup
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw = torch.tensor(0.0, device=device); sc2 = 0
                tgt = torch.tensor([p // BLOCK_SIZE for p in b['passkey_position']],
                                   device=device, dtype=torch.long)
                for _, inf in out['fixation_info'].items():
                    for lg in inf['fixation_logits']:
                        sw = sw + F.cross_entropy(lg, tgt.clamp(max=lg.shape[1]-1)); sc2 += 1
                if sc2: loss = loss + w * sw / sc2

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); step += 1

            if step % 100 == 0:
                log(f'  step {step} | loss {loss.item():.4f} | {elapsed:.0f}s')
        else:
            continue
        break

    # Final evaluation
    log('Evaluating on test set...')
    model.eval(); cor = tot = 0; dists = []
    with torch.no_grad():
        for b in test_ld:
            ids = b['input_ids'].to(device); out = model(ids)
            for _, inf in out['fixation_info'].items():
                for fp in inf['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(abs(fp[i].item() - b['passkey_position'][i]))
            for i in range(ids.shape[0]):
                pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                if pred == b['passkey'][i]: cor += 1
                tot += 1

    acc = cor / max(tot, 1)
    dist = sum(dists) / max(len(dists), 1)
    return acc, dist, step


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    model = QwenSaccadicPasskey().to(device)
    run_diagnostics(model, device, tokenizer)

    log('Starting training...')
    acc, dist, steps = train_and_eval(model, tokenizer, device)

    print(f'passkey_accuracy: {acc:.4f}')
    print(f'fixation_distance: {dist:.2f}')
    print(f'total_steps: {steps}')

if __name__ == '__main__':
    main()
