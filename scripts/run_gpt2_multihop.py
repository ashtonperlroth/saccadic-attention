"""GPT-2 Multi-Hop Passkey Experiment — Convergence-Based Training.

Trains until convergence for each (N_HOPS, N_SACCADES) config.
Convergence: val accuracy hasn't improved for 500 steps, or hits 100%, or 30 min cap.
Reports final accuracy AND time/steps to convergence.

2-hop results already established at 100% — noted but not re-run.
"""

import json
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 768; N_HEADS = 12; BLOCK_SIZE = 8; WINDOW_SIZE = 64
SACCADIC_LAYERS = [6, 7, 8, 9, 10, 11]; CONTEXT_LENGTH = 2048

LR = 5e-4; WEIGHT_DECAY = 0.01; GRAD_CLIP = 1.0; BATCH_SIZE = 4
ENTROPY_BONUS = 0.01; WARMUP_STEPS = 50
SUPERVISED_WARMUP_STEPS = 200; SUPERVISED_WARMUP_WEIGHT = 2.0
GUMBEL_TEMP_START = 1.0; GUMBEL_TEMP_END = 0.1; GUMBEL_ANNEAL_STEPS = 500

NUM_TRAIN = 3000; NUM_VAL = 100; NUM_EVAL = 200
MAX_TIME = 1800  # 30 min safety cap
PATIENCE = 500   # stop if no improvement for 500 steps
VAL_EVERY = 50   # check val accuracy every 50 steps

CONFIGS = [
    (3, 3), (3, 4),
    (4, 4), (4, 5),
    (5, 5),
]

def log(msg): print(msg, file=sys.stderr, flush=True)

_tok = GPT2Tokenizer.from_pretrained('gpt2')
ORDINALS = ['first', 'second', 'third', 'fourth', 'fifth']
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
    "Several factors contributed to the overall outcome.",
    "The meeting was scheduled for early in the morning.",
    "A small group discussed various approaches to the problem.",
    "The document outlined the key objectives clearly.",
    "Regular maintenance ensured smooth operation of equipment.",
    "The analysis revealed several interesting patterns.",
    "Participants shared their experiences and insights.",
    "The report summarized findings from the past quarter.",
    "Careful planning led to a successful implementation.",
    "The results exceeded expectations for the quarter.",
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiHopDataset(Dataset):
    def __init__(self, n, ctx_len, n_hops, seed=42):
        self.n_hops = n_hops
        self.rng = random.Random(seed)
        self.filler_enc = [_tok.encode(" " + s) for s in FILLER]
        self.prompt_ids = _tok.encode(f" What is the {n_hops}-digit code? The code is")
        self.samples = [self._make(i, ctx_len) for i in range(n)]

    def _make(self, idx, ctx_len):
        rng = random.Random(self.rng.randint(0, 2**32) + idx)
        digits = [rng.choice(string.digits) for _ in range(self.n_hops)]
        code = ''.join(digits)
        clues = [_tok.encode(f" The {ORDINALS[i]} digit of the code is {d}.")
                 for i, d in enumerate(digits)]
        answer_ids = _tok.encode(" " + code)
        clue_total = sum(len(c) for c in clues)
        budget = ctx_len - clue_total - len(self.prompt_ids) - len(answer_ids)
        fill = []
        while len(fill) < budget:
            fill.extend(rng.choice(self.filler_enc))
        fill = fill[:budget]

        seg = len(fill) // (self.n_hops + 1)
        clue_positions = []
        full = list(fill)
        for i in range(self.n_hops - 1, -1, -1):
            pos = min(i * seg + rng.randint(0, seg), len(full))
            clue_positions.insert(0, pos)
            full = full[:pos] + clues[i] + full[pos:]

        full = full + self.prompt_ids + answer_ids
        if len(full) > ctx_len: full = full[:ctx_len]
        elif len(full) < ctx_len: full += [_tok.eos_token_id] * (ctx_len - len(full))

        return (torch.tensor(full[:ctx_len], dtype=torch.long),
                torch.tensor([int(d) for d in digits], dtype=torch.long),
                code, clue_positions)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids, dl, code, cps = self.samples[i]
        return {'input_ids': ids, 'digit_labels': dl, 'passkey': code,
                'passkey_position': cps[0], 'clue_positions': cps}

def collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'digit_labels': torch.stack([b['digit_labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
        'clue_positions': [b['clue_positions'] for b in batch],
    }


# ── Saccadic Components ──────────────────────────────────────────────────────

class PeripheralEncoder(nn.Module):
    def __init__(self, dim, bs):
        super().__init__()
        self.bs = bs; self.w = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim * 3, dim); self.norm = nn.LayerNorm(dim)
        self.pos = nn.Embedding(16384, dim)
    def forward(self, x):
        B, N, D = x.shape
        pad = (self.bs - N % self.bs) % self.bs
        if pad: x = F.pad(x, (0, 0, 0, pad))
        nb = x.shape[1] // self.bs
        blk = x.reshape(B, nb, self.bs, D)
        wt = F.softmax(self.w(blk).squeeze(-1), dim=-1)
        mn = torch.einsum('bnk,bnkd->bnd', wt, blk)
        sd = (torch.einsum('bnk,bnkd->bnd', wt, (blk - mn.unsqueeze(2))**2) + 1e-8).sqrt()
        mx = blk.max(dim=2).values
        out = self.norm(self.proj(torch.cat([mn, sd, mx], -1)))
        return out + self.pos(torch.arange(nb, device=x.device))

class Controller(nn.Module):
    def __init__(self, dim, bs):
        super().__init__()
        self.bs, self.dim = bs, dim
        self.qp = nn.Linear(dim, dim); self.kp = nn.Linear(dim, dim)
        self.temperature = GUMBEL_TEMP_START
    def forward(self, pm, state):
        s = torch.einsum('bd,bmd->bm', self.qp(state), self.kp(pm)) / math.sqrt(self.dim)
        if self.training:
            sel = F.gumbel_softmax(s, tau=self.temperature, hard=True)
            idx = torch.einsum('bm,m->b', sel, torch.arange(pm.shape[1], device=s.device, dtype=torch.float))
        else:
            idx = s.argmax(-1).float()
        return (idx * self.bs).long(), s, idx.long()

class Foveal(nn.Module):
    def __init__(self, dim, nh, ws):
        super().__init__()
        self.ws = ws
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def _ext(self, x, fp):
        B, N, D = x.shape; h = self.ws // 2; ws = []
        for i in range(B):
            c = fp[i].item(); s, e = max(0, c-h), min(N, c+h)
            if e-s < self.ws:
                s = max(0, e-self.ws) if s > 0 else 0; e = min(N, s+self.ws)
            w = x[i, s:e]
            if w.shape[0] < self.ws: w = F.pad(w, (0, 0, 0, self.ws - w.shape[0]))
            ws.append(w)
        return torch.stack(ws)
    def forward(self, x, fp, state, acc=None):
        win = self._ext(x, fp)
        ctx = torch.cat([state.unsqueeze(1)] + ([acc, win] if acc is not None else [win]), 1)
        out, _ = self.attn(self.n1(ctx), self.n1(ctx), self.n1(ctx))
        ctx = ctx + out; cl = ctx[:, 0]
        cl = cl + self.ff(self.n2(cl))
        return cl, (torch.cat([acc, win], 1) if acc is not None else win)

class SaccadicBlock(nn.Module):
    def __init__(self, orig, dim, nh, n_saccades):
        super().__init__()
        self.ln_1 = orig.ln_1; self.ln_2 = orig.ln_2; self.mlp = orig.mlp
        self.n_saccades = n_saccades
        self.pe = PeripheralEncoder(dim, BLOCK_SIZE)
        self.ctrl = Controller(dim, BLOCK_SIZE)
        self.fov = Foveal(dim, nh, WINDOW_SIZE)
        self.ma = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.mn = nn.LayerNorm(dim)
        self.mg = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.op = nn.Linear(dim, dim); self.on = nn.LayerNorm(dim)
    def forward(self, x, ps=None):
        B, N, D = x.shape; res = x; h = self.ln_1(x)
        pm = self.pe(ps if ps is not None else h); st = pm.mean(1)
        fps, fls = [], []; ac = None
        for t in range(self.n_saccades):
            fp, lg, _ = self.ctrl(pm, st); fps.append(fp); fls.append(lg)
            st, ac = self.fov(h, fp, st, ac)
            d, _ = self.ma(self.mn(pm), ac, ac)
            a = self.mg(torch.tensor([[t / self.n_saccades]], device=x.device, dtype=x.dtype))
            pm = pm + a * d
        x = res + self.op(self.on(st.unsqueeze(1).expand(-1, N, -1)))
        x = x + self.mlp(self.ln_2(x))
        return x, {'fixation_points': fps, 'fixation_logits': fls}


# ── Model ─────────────────────────────────────────────────────────────────────

class GPT2MultiHop(nn.Module):
    def __init__(self, n_hops, n_saccades):
        super().__init__()
        self.n_hops = n_hops
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        dim = self.gpt2.config.n_embd; nh = self.gpt2.config.n_head
        self.max_pos = self.gpt2.config.n_positions
        self.sblocks = nn.ModuleDict()
        for i in SACCADIC_LAYERS:
            self.sblocks[str(i)] = SaccadicBlock(self.gpt2.transformer.h[i], dim, nh, n_saccades)
        for p in self.gpt2.parameters(): p.requires_grad = False
        for b in self.sblocks.values():
            for m in [b.pe, b.ctrl, b.fov, b.ma, b.mn, b.mg, b.op, b.on]:
                for pp in m.parameters(): pp.requires_grad = True
        self.digit_heads = nn.ModuleList([nn.Linear(dim, 10) for _ in range(n_hops)])
        self.first_sacc = min(SACCADIC_LAYERS)

    def forward(self, ids, labels=None):
        B, N = ids.shape; dev = ids.device
        pos = torch.arange(N, device=dev).clamp(max=self.max_pos - 1).unsqueeze(0)
        x = self.gpt2.transformer.drop(self.gpt2.transformer.wte(ids) + self.gpt2.transformer.wpe(pos))
        cp = torch.arange(N, device=dev); ps = None; info = {}
        for i, blk in enumerate(self.gpt2.transformer.h):
            if str(i) in self.sblocks:
                x, inf = self.sblocks[str(i)](x, ps); info[i] = inf
            else:
                x = blk(x, cache_position=cp)
                if i == self.first_sacc - 1: ps = x.detach()
        x = self.gpt2.transformer.ln_f(x); last = x[:, -1]
        dl = [h(last) for h in self.digit_heads]
        loss = sum(F.cross_entropy(dl[i], labels[:, i]) for i in range(self.n_hops)) / self.n_hops if labels is not None else None
        return {'loss': loss, 'digit_logits': dl, 'fixation_info': info}

    def set_gumbel_temperature(self, t):
        for b in self.sblocks.values(): b.ctrl.temperature = t

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick Validation ──────────────────────────────────────────────────────────

def quick_val(model, val_loader, n_hops, device):
    """Fast accuracy check on validation set."""
    model.eval()
    cor = tot = 0
    with torch.no_grad():
        for b in val_loader:
            ids = b['input_ids'].to(device)
            out = model(ids)
            for i in range(ids.shape[0]):
                pred = ''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred == b['passkey'][i]: cor += 1
                tot += 1
    model.train()
    return cor / max(tot, 1)


# ── Training with Convergence ─────────────────────────────────────────────────

def gt(step):
    p = min(step / max(GUMBEL_ANNEAL_STEPS, 1), 1.0)
    return GUMBEL_TEMP_START + (GUMBEL_TEMP_END - GUMBEL_TEMP_START) * p


def train_until_converged(model, n_hops, device):
    """Train until val accuracy plateaus, hits 100%, or 30 min cap."""
    train_ds = MultiHopDataset(NUM_TRAIN, CONTEXT_LENGTH, n_hops, seed=42)
    val_ds = MultiHopDataset(NUM_VAL, CONTEXT_LENGTH, n_hops, seed=77777)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    def lrf(s):
        if s < WARMUP_STEPS: return s / max(WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP_STEPS) / 5000))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrf)

    model.train()
    t0 = time.time()
    step = 0
    best_val_acc = 0.0
    steps_since_improvement = 0
    converge_step = 0
    converge_reason = 'max_time'

    while True:
        for b in train_ld:
            elapsed = time.time() - t0

            # Check stopping conditions
            if elapsed >= MAX_TIME:
                converge_reason = f'max_time ({MAX_TIME}s)'
                log(f'  MAX TIME reached: {step} steps, {elapsed:.0f}s, best_val={best_val_acc:.4f}')
                return step, elapsed, best_val_acc, converge_reason

            # Periodic validation
            if step > 0 and step % VAL_EVERY == 0:
                val_acc = quick_val(model, val_ld, n_hops, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    steps_since_improvement = 0
                    converge_step = step
                    log(f'  step {step} | val_acc={val_acc:.4f} (NEW BEST) | {elapsed:.0f}s')
                    if val_acc >= 1.0:
                        converge_reason = 'perfect_accuracy'
                        log(f'  PERFECT ACCURACY at step {step}, {elapsed:.0f}s')
                        return step, elapsed, best_val_acc, converge_reason
                else:
                    steps_since_improvement += VAL_EVERY
                    if step % 200 == 0:
                        log(f'  step {step} | val_acc={val_acc:.4f} (best={best_val_acc:.4f}, plateau={steps_since_improvement}) | {elapsed:.0f}s')

                if steps_since_improvement >= PATIENCE:
                    converge_reason = f'plateau ({PATIENCE} steps no improvement)'
                    log(f'  PLATEAU at step {step}, best_val={best_val_acc:.4f}, {elapsed:.0f}s')
                    return step, elapsed, best_val_acc, converge_reason

            # Training step
            ids = b['input_ids'].to(device)
            dl = b['digit_labels'].to(device)
            model.set_gumbel_temperature(gt(step))

            out = model(ids, labels=dl)
            loss = out['loss']

            # Entropy bonus
            te = torch.tensor(0.0); ec = 0
            for _, inf in out['fixation_info'].items():
                for lg in inf['fixation_logits']:
                    p = F.softmax(lg, -1)
                    te = te + (-(p * (p + 1e-8).log()).sum(-1).mean()).to(te.device); ec += 1
            if ec: loss = loss - ENTROPY_BONUS * te / ec

            # Supervised warmup
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw = torch.tensor(0.0, device=device); sc2 = 0
                for _, inf in out['fixation_info'].items():
                    for si, lg in enumerate(inf['fixation_logits']):
                        ci = si % n_hops
                        tgt = torch.tensor(
                            [cp[ci] // BLOCK_SIZE for cp in b['clue_positions']],
                            device=device, dtype=torch.long).clamp(max=lg.shape[1] - 1)
                        sw = sw + F.cross_entropy(lg, tgt); sc2 += 1
                if sc2: loss = loss + w * sw / sc2

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step(); sched.step(); step += 1


# ── Final Evaluation ──────────────────────────────────────────────────────────

def final_eval(model, n_hops, device):
    ds = MultiHopDataset(NUM_EVAL, CONTEXT_LENGTH, n_hops, seed=99999)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    cor = tot = 0; dists = []
    with torch.no_grad():
        for b in ld:
            ids = b['input_ids'].to(device); out = model(ids)
            for _, inf in out['fixation_info'].items():
                for fp in inf['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.extend([min(abs(fp[i].item() - cp) for cp in b['clue_positions'][i])])
            for i in range(ids.shape[0]):
                pred = ''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred == b['passkey'][i]: cor += 1
                tot += 1
    return cor / max(tot, 1), sum(dists) / max(len(dists), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    # Note 2-hop results from previous run (already 100%)
    results = [
        {'n_hops': 2, 'n_saccades': 2, 'accuracy': 1.0, 'avg_clue_distance': 330.03,
         'steps': 'N/A', 'time_s': 'N/A', 'converge_reason': 'previous_run'},
        {'n_hops': 2, 'n_saccades': 3, 'accuracy': 1.0, 'avg_clue_distance': 210.77,
         'steps': 'N/A', 'time_s': 'N/A', 'converge_reason': 'previous_run'},
    ]

    for n_hops, n_saccades in CONFIGS:
        log(f'\n{"="*60}')
        log(f'N_HOPS={n_hops}, N_SACCADES={n_saccades} — training until convergence')
        log(f'{"="*60}')

        model = GPT2MultiHop(n_hops, n_saccades).to(device)
        log(f'Trainable: {model.trainable_params():,}')

        steps, elapsed, best_val, reason = train_until_converged(model, n_hops, device)
        acc, dist = final_eval(model, n_hops, device)

        log(f'FINAL: accuracy={acc:.4f}, distance={dist:.2f}, '
            f'steps={steps}, time={elapsed:.0f}s, reason={reason}')

        results.append({
            'n_hops': n_hops, 'n_saccades': n_saccades,
            'accuracy': acc, 'avg_clue_distance': dist,
            'steps': steps, 'time_s': round(elapsed, 1),
            'converge_reason': reason,
        })

        del model; torch.cuda.empty_cache()

    # Print results
    print('n_hops\tn_saccades\taccuracy\tavg_clue_distance\tsteps\ttime_s\tconverge_reason')
    for r in results:
        print(f'{r["n_hops"]}\t{r["n_saccades"]}\t{r["accuracy"]:.4f}\t'
              f'{r["avg_clue_distance"]:.2f}\t{r["steps"]}\t{r["time_s"]}\t{r["converge_reason"]}')

    with open('gpt2_multihop_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
