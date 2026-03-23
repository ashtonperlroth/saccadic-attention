"""Tiny saccadic transformer trained from scratch on passkey retrieval.

Single-file experiment script (Karpathy style). The autoresearch agent edits
THIS FILE. Everything is fair game except the evaluation section.

Prints exactly two lines to stdout:
    passkey_accuracy: X.XXXX
    fixation_distance: X.XX
"""

import math
import random
import string
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ── Hyperparameters (EDIT THESE) ──────────────────────────────────────────────

# Model
N_LAYERS = 4
HIDDEN_DIM = 128
N_HEADS = 4
SACCADIC_LAYERS = [2, 3]
VOCAB_SIZE = 512                   # small char-level vocab
MAX_POS = 2048

# Saccadic
NUM_SACCADES = 3
WINDOW_SIZE = 64
BLOCK_SIZE = 8

# Gumbel
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_END = 0.1
GUMBEL_ANNEAL_STEPS = 800

# Training
BATCH_SIZE = 16
LR = 2e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ENTROPY_BONUS = 0.01
WARMUP_STEPS = 100
SUPERVISED_WARMUP_STEPS = 300
SUPERVISED_WARMUP_WEIGHT = 2.0

# Data
TRAIN_CONTEXT_LENGTH = 2048  # can be reduced for curriculum learning
EVAL_CONTEXT_LENGTH = 2048   # FIXED. Do not change.
NUM_TRAIN_SAMPLES = 5000
NUM_EVAL_SAMPLES = 200

# Budget
WALL_CLOCK_BUDGET = 300  # FIXED 5 minutes. Do not change.


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Tokenizer (dead simple) ──────────────────────────────────────────────────

# Fixed character-level vocab: digits, lowercase, uppercase, punctuation, space
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' \n'
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = padding
ID_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
PAD_ID = 0
assert len(CHAR_TO_ID) < VOCAB_SIZE


def encode(text):
    return [CHAR_TO_ID.get(c, CHAR_TO_ID[' ']) for c in text]


def decode(ids):
    return ''.join(ID_TO_CHAR.get(i, ' ') for i in ids if i != PAD_ID)


# ── Dataset ───────────────────────────────────────────────────────────────────

FILLER_SENTENCES = [
    "the weather was pleasant and the sky was clear. ",
    "several researchers gathered to discuss the latest findings. ",
    "the library contained thousands of books on various topics. ",
    "traffic moved slowly through the busy intersection. ",
    "a gentle breeze rustled through the autumn leaves. ",
    "the project deadline was approaching rapidly. ",
    "students worked diligently on their assignments. ",
    "the old building stood at the corner of the street. ",
    "new developments in technology continued to emerge. ",
    "the garden was well maintained throughout the year. ",
    "several factors contributed to the overall outcome. ",
    "the meeting was scheduled for early in the morning. ",
    "a small group discussed various approaches to the problem. ",
    "the document outlined the key objectives clearly. ",
    "regular maintenance ensured smooth operation of equipment. ",
    "the analysis revealed several interesting patterns. ",
    "participants shared their experiences and insights. ",
    "the report summarized findings from the past quarter. ",
    "careful planning led to a successful implementation. ",
    "the results exceeded expectations for the quarter. ",
]


class PasskeyDataset(Dataset):
    def __init__(self, n_samples, ctx_len, seed=42):
        self.n = n_samples
        self.ctx_len = ctx_len
        self.rng = random.Random(seed)
        # Pre-encode filler sentences
        self.filler_encoded = [encode(s) for s in FILLER_SENTENCES]
        self.prompt_ids = encode(" what is the secret number? the secret number is ")
        self.samples = [self._make(i) for i in range(n_samples)]

    def _make(self, idx):
        rng = random.Random(self.rng.randint(0, 2**32) + idx)
        passkey = ''.join(rng.choices(string.digits, k=5))
        passkey_ids = encode(f" the secret number is {passkey}. ")
        answer_ids = encode(passkey)

        reserved = len(passkey_ids) + len(self.prompt_ids) + len(answer_ids)
        filler_budget = self.ctx_len - reserved

        filler = []
        while len(filler) < filler_budget:
            filler.extend(rng.choice(self.filler_encoded))
        filler = filler[:filler_budget]

        insert_pos = rng.randint(0, len(filler))
        passkey_char_pos = insert_pos

        full = filler[:insert_pos] + passkey_ids + filler[insert_pos:] + self.prompt_ids + answer_ids
        if len(full) > self.ctx_len:
            full = full[:self.ctx_len]
        elif len(full) < self.ctx_len:
            full = full + [PAD_ID] * (self.ctx_len - len(full))

        input_ids = torch.tensor(full, dtype=torch.long)
        # Labels: 5 digit targets (0-9) for the classification head
        digit_labels = torch.tensor([int(d) for d in passkey], dtype=torch.long)

        return input_ids, digit_labels, passkey, passkey_char_pos

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input_ids, digit_labels, passkey, pos = self.samples[idx]
        return {'input_ids': input_ids, 'digit_labels': digit_labels, 'passkey': passkey, 'passkey_position': pos}


def collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'digit_labels': torch.stack([b['digit_labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
    }


# ── Model Components ──────────────────────────────────────────────────────────

class PeripheralEncoder(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.weight_proj = nn.Linear(dim, 1)
        self.stats_proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.pos_emb = nn.Embedding(16384, dim)

    def forward(self, x):
        B, N, D = x.shape
        # Pad to multiple of block_size
        pad = (self.block_size - N % self.block_size) % self.block_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        n_blocks = x.shape[1] // self.block_size
        # (B, n_blocks, block_size, D)
        blocks = x.reshape(B, n_blocks, self.block_size, D)
        # Weighted mean
        w = self.weight_proj(blocks).squeeze(-1)          # (B, nb, bs)
        w = F.softmax(w, dim=-1)
        wmean = torch.einsum('bnk,bnkd->bnd', w, blocks)  # (B, nb, D)
        # Std
        diff = blocks - wmean.unsqueeze(2)
        wvar = torch.einsum('bnk,bnkd->bnd', w, diff ** 2)
        wstd = (wvar + 1e-8).sqrt()
        # Max pool
        bmax = blocks.max(dim=2).values
        # Project concat(mean, std, max) -> D
        out = self.stats_proj(torch.cat([wmean, wstd, bmax], dim=-1))
        out = self.norm(out)
        pos = torch.arange(n_blocks, device=x.device)
        out = out + self.pos_emb(pos)
        return out  # (B, n_blocks, D)


class SaccadicController(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.temperature = GUMBEL_TEMP_START

    def forward(self, periph_map, state, history=None, mask=None):
        q = self.q_proj(state)                                        # (B, D)
        k = self.k_proj(periph_map)                                   # (B, M, D)
        scores = torch.einsum('bd,bmd->bm', q, k) / math.sqrt(self.dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        logits = scores
        if self.training:
            sel = F.gumbel_softmax(scores, tau=self.temperature, hard=True)
            idx_f = torch.einsum('bm,m->b', sel, torch.arange(periph_map.shape[1], device=scores.device, dtype=torch.float))
        else:
            idx_f = scores.argmax(dim=-1).float()
        fix_pt = (idx_f * self.block_size).long()
        block_idx = idx_f.long()
        return fix_pt, logits, block_idx


class FovealProcessor(nn.Module):
    def __init__(self, dim, n_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def _extract(self, x, fix_pt):
        B, N, D = x.shape
        half = self.window_size // 2
        wins = []
        for i in range(B):
            c = fix_pt[i].item()
            s = max(0, c - half)
            e = min(N, c + half)
            if e - s < self.window_size:
                s = max(0, e - self.window_size) if s > 0 else 0
                e = min(N, s + self.window_size)
            w = x[i, s:e]
            if w.shape[0] < self.window_size:
                w = F.pad(w, (0, 0, 0, self.window_size - w.shape[0]))
            wins.append(w)
        return torch.stack(wins)

    def forward(self, x, fix_pt, state, acc_ctx=None):
        window = self._extract(x, fix_pt)           # (B, ws, D)
        cls = state.unsqueeze(1)                     # (B, 1, D)
        if acc_ctx is not None:
            ctx = torch.cat([cls, acc_ctx, window], dim=1)
        else:
            ctx = torch.cat([cls, window], dim=1)
        normed = self.norm1(ctx)
        out, _ = self.attn(normed, normed, normed)
        ctx = ctx + out
        cls_out = ctx[:, 0]
        cls_out = cls_out + self.ffn(self.norm2(cls_out))
        new_acc = torch.cat([acc_ctx, window], dim=1) if acc_ctx is not None else window
        return cls_out, new_acc


class SaccadicAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.periph = PeripheralEncoder(dim, BLOCK_SIZE)
        self.controller = SaccadicController(dim, BLOCK_SIZE)
        self.foveal = FovealProcessor(dim, n_heads, WINDOW_SIZE)
        # Peripheral map update
        self.map_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.map_norm = nn.LayerNorm(dim)
        self.map_gate = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        # Output
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        # Block
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, periph_src=None):
        B, N, D = x.shape
        residual = x
        h = self.ln1(x)
        src = periph_src if periph_src is not None else h
        pmap = self.periph(src)
        state = pmap.mean(dim=1)
        fix_pts, fix_logits_all, fix_history = [], [], []
        acc_ctx = None
        for t in range(NUM_SACCADES):
            fp, logits, bidx = self.controller(pmap, state, fix_history)
            fix_pts.append(fp)
            fix_logits_all.append(logits)
            fix_history.append(bidx)
            state, acc_ctx = self.foveal(h, fp, state, acc_ctx)
            # Peripheral map update
            normed = self.map_norm(pmap)
            delta, _ = self.map_attn(normed, acc_ctx, acc_ctx)
            step_in = torch.tensor([[t / NUM_SACCADES]], device=x.device, dtype=x.dtype)
            alpha = self.map_gate(step_in)
            pmap = pmap + alpha * delta
        # Broadcast state back
        out = self.out_proj(self.out_norm(state.unsqueeze(1).expand(-1, N, -1)))
        x = residual + out
        # FFN
        residual = x
        x = residual + self.mlp(self.ln2(x))
        info = {'fixation_points': fix_pts, 'fixation_logits': fix_logits_all}
        return x, info


class StandardAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        N = x.shape[1]
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        residual = x
        h = self.ln1(x)
        out, _ = self.attn(h, h, h, attn_mask=mask)
        x = residual + out
        residual = x
        x = residual + self.mlp(self.ln2(x))
        return x


class TinySaccadicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_emb = nn.Embedding(MAX_POS, HIDDEN_DIM)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        self.saccadic_layer_indices = set(SACCADIC_LAYERS)
        for i in range(N_LAYERS):
            if i in self.saccadic_layer_indices:
                self.layers.append(SaccadicAttentionLayer(HIDDEN_DIM, N_HEADS))
            else:
                self.layers.append(StandardAttentionLayer(HIDDEN_DIM, N_HEADS))
        self.ln_f = nn.LayerNorm(HIDDEN_DIM)
        # Digit classifier: 5 digits × 10 classes each
        # Uses the last token's representation (after the prompt)
        self.digit_heads = nn.ModuleList([nn.Linear(HIDDEN_DIM, 10) for _ in range(5)])
        self._n_params = sum(p.numel() for p in self.parameters())
        self.first_saccadic = min(SACCADIC_LAYERS)

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        periph_src = None
        all_info = {}
        for i, layer in enumerate(self.layers):
            if i in self.saccadic_layer_indices:
                x, info = layer(x, periph_src=periph_src)
                all_info[i] = info
            else:
                x = layer(x)
                if i == self.first_saccadic - 1:
                    periph_src = x.detach()
        x = self.ln_f(x)
        # Use the last token's representation to classify all 5 digits
        last_hidden = x[:, -1, :]  # (B, D)
        # digit_logits: list of 5 tensors, each (B, 10)
        digit_logits = [head(last_hidden) for head in self.digit_heads]
        loss = None
        if labels is not None:
            # labels is a (B, 5) tensor of digit targets (0-9)
            loss = sum(F.cross_entropy(digit_logits[i], labels[:, i]) for i in range(5)) / 5
        return {'loss': loss, 'digit_logits': digit_logits, 'fixation_info': all_info}

    def set_gumbel_temperature(self, temp):
        for layer in self.layers:
            if isinstance(layer, SaccadicAttentionLayer):
                layer.controller.temperature = temp


# ── Training ──────────────────────────────────────────────────────────────────

def get_gumbel_temp(step):
    progress = min(step / max(GUMBEL_ANNEAL_STEPS, 1), 1.0)
    return GUMBEL_TEMP_START + (GUMBEL_TEMP_END - GUMBEL_TEMP_START) * progress


def supervised_warmup_loss(fixation_info, passkey_positions, device):
    total = torch.tensor(0.0, device=device)
    count = 0
    targets = torch.tensor([p // BLOCK_SIZE for p in passkey_positions], device=device, dtype=torch.long)
    for layer_idx, info in fixation_info.items():
        for logits in info['fixation_logits']:
            n_blocks = logits.shape[1]
            t = targets.clamp(max=n_blocks - 1)
            total = total + F.cross_entropy(logits, t)
            count += 1
    return total / max(count, 1)


def entropy_bonus(fixation_info):
    total = torch.tensor(0.0)
    count = 0
    for _, info in fixation_info.items():
        for logits in info['fixation_logits']:
            p = F.softmax(logits, dim=-1)
            ent = -(p * (p + 1e-8).log()).sum(-1).mean()
            total = total + ent.to(total.device)
            count += 1
    return total / max(count, 1)


def train(model, device):
    ds = PasskeyDataset(NUM_TRAIN_SAMPLES, TRAIN_CONTEXT_LENGTH, seed=42)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Linear warmup then cosine decay
    def lr_fn(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / max(5000 - WARMUP_STEPS, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    model.train()
    t0 = time.time()
    step = 0
    budget = WALL_CLOCK_BUDGET * 0.80  # reserve 20% for eval

    while True:
        for batch in loader:
            if time.time() - t0 >= budget:
                log(f'Budget reached: {step} steps in {time.time()-t0:.0f}s')
                return
            ids = batch['input_ids'].to(device)
            dlabs = batch['digit_labels'].to(device)
            model.set_gumbel_temperature(get_gumbel_temp(step))

            out = model(ids, labels=dlabs)
            loss = out['loss']
            # Entropy bonus
            ent = entropy_bonus(out['fixation_info'])
            loss = loss - ENTROPY_BONUS * ent
            # Supervised warmup
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw = supervised_warmup_loss(out['fixation_info'], batch['passkey_position'], device)
                loss = loss + w * sw

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            step += 1

            if step % 100 == 0:
                log(f'  step {step} | loss {loss.item():.4f} | lm {out["loss"].item():.4f} | '
                    f'temp {get_gumbel_temp(step):.3f} | {time.time()-t0:.0f}s')


# ── Evaluation (DO NOT MODIFY) ───────────────────────────────────────────────

def evaluate(model, device):
    ds = PasskeyDataset(NUM_EVAL_SAMPLES, EVAL_CONTEXT_LENGTH, seed=99999)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    correct = total = 0
    dists = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            out = model(ids)
            digit_logits = out['digit_logits']  # list of 5 × (B, 10)
            for layer_idx, info in out['fixation_info'].items():
                for fp in info['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(abs(fp[i].item() - batch['passkey_position'][i]))
            # Check if all 5 predicted digits match the passkey
            for i in range(ids.shape[0]):
                pred_digits = ''.join(str(dl[i].argmax().item()) for dl in digit_logits)
                if pred_digits == batch['passkey'][i]:
                    correct += 1
                total += 1
    acc = correct / max(total, 1)
    dist = sum(dists) / max(len(dists), 1)
    return acc, dist


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    model = TinySaccadicTransformer().to(device)
    log(f'Parameters: {model._n_params:,}')
    log(f'Training for up to {WALL_CLOCK_BUDGET}s...')
    train(model, device)
    log('Evaluating...')
    acc, dist = evaluate(model, device)
    print(f'passkey_accuracy: {acc:.4f}')
    print(f'fixation_distance: {dist:.2f}')


if __name__ == '__main__':
    main()
