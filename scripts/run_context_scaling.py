"""Context length scaling: train once at 2048, evaluate at 1024-16384.

Tests whether saccadic attention generalizes to unseen context lengths.
The saccadic layers see more peripheral blocks at longer contexts but
the compute per saccade stays fixed (3 × 64² = 12,288 ops).

Uses local window attention (256 tokens) in layers 0-1 to avoid OOM.
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

# ── Config (T15 winning) ─────────────────────────────────────────────────────

HIDDEN_DIM = 128
N_HEADS = 4
NUM_SACCADES = 3
WINDOW_SIZE = 64
BLOCK_SIZE = 8
VOCAB_SIZE = 512
BATCH_SIZE = 16
LR = 2e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ENTROPY_BONUS = 0.01
WARMUP_STEPS = 100
SUPERVISED_WARMUP_STEPS = 300
SUPERVISED_WARMUP_WEIGHT = 2.0
GUMBEL_TEMP_START = 1.0
GUMBEL_TEMP_END = 0.1
GUMBEL_ANNEAL_STEPS = 800
LOCAL_WINDOW = 256

TRAIN_CTX = 2048
EVAL_CONTEXTS = [1024, 2048, 4096, 8192, 16384]
NUM_TRAIN = 5000
NUM_EVAL = 200
WALL_CLOCK_BUDGET = 300


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' \n'
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}
PAD_ID = 0

def encode(text):
    return [CHAR_TO_ID.get(c, CHAR_TO_ID[' ']) for c in text]


# ── Dataset ───────────────────────────────────────────────────────────────────

FILLER = [
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
    def __init__(self, n, ctx_len, seed=42):
        self.rng = random.Random(seed)
        self.filler_enc = [encode(s) for s in FILLER]
        self.prompt = encode(" what is the secret number? the secret number is ")
        self.samples = [self._make(i, ctx_len) for i in range(n)]

    def _make(self, idx, ctx_len):
        rng = random.Random(self.rng.randint(0, 2**32) + idx)
        pk = ''.join(rng.choices(string.digits, k=5))
        pk_ids = encode(f" the secret number is {pk}. ")
        ans = encode(pk)
        budget = ctx_len - len(pk_ids) - len(self.prompt) - len(ans)
        fill = []
        while len(fill) < budget:
            fill.extend(rng.choice(self.filler_enc))
        fill = fill[:budget]
        pos = rng.randint(0, len(fill))
        full = fill[:pos] + pk_ids + fill[pos:] + self.prompt + ans
        if len(full) > ctx_len: full = full[:ctx_len]
        elif len(full) < ctx_len: full += [PAD_ID] * (ctx_len - len(full))
        return (torch.tensor(full, dtype=torch.long),
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


# ── Model ─────────────────────────────────────────────────────────────────────

class LocalAttentionLayer(nn.Module):
    """Causal attention restricted to a local window. O(n·w) not O(n²)."""
    def __init__(self, dim, n_heads, window):
        super().__init__()
        self.window = window
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        h = self.ln1(x)
        row = torch.arange(N, device=x.device)
        col = torch.arange(N, device=x.device)
        # mask[i,j]=True means BLOCKED: future tokens + tokens beyond window
        mask = (col.unsqueeze(0) > row.unsqueeze(1)) | \
               (col.unsqueeze(0) < (row.unsqueeze(1) - self.window + 1))
        out, _ = self.attn(h, h, h, attn_mask=mask)
        x = residual + out
        x = x + self.mlp(self.ln2(x))
        return x


class PeripheralEncoder(nn.Module):
    def __init__(self, dim, bs):
        super().__init__()
        self.bs = bs
        self.w = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim*3, dim)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Embedding(16384, dim)

    def forward(self, x):
        B, N, D = x.shape
        pad = (self.bs - N % self.bs) % self.bs
        if pad: x = F.pad(x, (0,0,0,pad))
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
        self.qp = nn.Linear(dim, dim)
        self.kp = nn.Linear(dim, dim)
        self.temperature = GUMBEL_TEMP_START

    def forward(self, pm, state):
        s = torch.einsum('bd,bmd->bm', self.qp(state), self.kp(pm)) / math.sqrt(self.dim)
        logits = s
        if self.training:
            sel = F.gumbel_softmax(s, tau=self.temperature, hard=True)
            idx = torch.einsum('bm,m->b', sel, torch.arange(pm.shape[1], device=s.device, dtype=torch.float))
        else:
            idx = s.argmax(-1).float()
        return (idx * self.bs).long(), logits, idx.long()


class Foveal(nn.Module):
    def __init__(self, dim, nh, ws):
        super().__init__()
        self.ws = ws
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def _ext(self, x, fp):
        B, N, D = x.shape
        h = self.ws // 2
        ws = []
        for i in range(B):
            c = fp[i].item()
            s, e = max(0, c-h), min(N, c+h)
            if e-s < self.ws:
                s = max(0, e-self.ws) if s > 0 else 0
                e = min(N, s+self.ws)
            w = x[i, s:e]
            if w.shape[0] < self.ws: w = F.pad(w, (0,0,0,self.ws-w.shape[0]))
            ws.append(w)
        return torch.stack(ws)

    def forward(self, x, fp, state, acc=None):
        win = self._ext(x, fp)
        ctx = torch.cat([state.unsqueeze(1)] + ([acc, win] if acc is not None else [win]), 1)
        out, _ = self.attn(self.n1(ctx), self.n1(ctx), self.n1(ctx))
        ctx = ctx + out
        cl = ctx[:, 0]
        cl = cl + self.ff(self.n2(cl))
        nacc = torch.cat([acc, win], 1) if acc is not None else win
        return cl, nacc


class SaccLayer(nn.Module):
    def __init__(self, dim, nh, bs):
        super().__init__()
        self.pe = PeripheralEncoder(dim, bs)
        self.ctrl = Controller(dim, bs)
        self.fov = Foveal(dim, nh, WINDOW_SIZE)
        self.ma = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.mn = nn.LayerNorm(dim)
        self.mg = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.op = nn.Linear(dim, dim)
        self.on = nn.LayerNorm(dim)
        self.l1 = nn.LayerNorm(dim)
        self.l2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x, ps=None):
        B, N, D = x.shape
        res = x
        h = self.l1(x)
        pm = self.pe(ps if ps is not None else h)
        st = pm.mean(1)
        fps, fls = [], []
        ac = None
        for t in range(NUM_SACCADES):
            fp, lg, _ = self.ctrl(pm, st)
            fps.append(fp); fls.append(lg)
            st, ac = self.fov(h, fp, st, ac)
            d, _ = self.ma(self.mn(pm), ac, ac)
            a = self.mg(torch.tensor([[t/NUM_SACCADES]], device=x.device, dtype=x.dtype))
            pm = pm + a * d
        x = res + self.op(self.on(st.unsqueeze(1).expand(-1, N, -1)))
        x = x + self.mlp(self.l2(x))
        return x, {'fixation_points': fps, 'fixation_logits': fls}


class Model(nn.Module):
    def __init__(self, max_pos):
        super().__init__()
        self.te = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(max_pos, HIDDEN_DIM)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            LocalAttentionLayer(HIDDEN_DIM, N_HEADS, LOCAL_WINDOW),
            LocalAttentionLayer(HIDDEN_DIM, N_HEADS, LOCAL_WINDOW),
            SaccLayer(HIDDEN_DIM, N_HEADS, BLOCK_SIZE),
            SaccLayer(HIDDEN_DIM, N_HEADS, BLOCK_SIZE),
        ])
        self.lnf = nn.LayerNorm(HIDDEN_DIM)
        self.heads = nn.ModuleList([nn.Linear(HIDDEN_DIM, 10) for _ in range(5)])
        self._np = sum(p.numel() for p in self.parameters())
        self.max_pos = max_pos

    def forward(self, ids, labels=None):
        B, N = ids.shape
        # Clamp position ids to max_pos for generalization to longer contexts
        pos = torch.arange(N, device=ids.device).clamp(max=self.max_pos - 1)
        x = self.drop(self.te(ids) + self.pe(pos))
        ps = None
        info = {}
        for i, l in enumerate(self.layers):
            if isinstance(l, SaccLayer):
                x, inf = l(x, ps)
                info[i] = inf
            else:
                x = l(x)
                if i == 1: ps = x.detach()
        x = self.lnf(x)
        dl = [h(x[:, -1]) for h in self.heads]
        loss = sum(F.cross_entropy(dl[i], labels[:, i]) for i in range(5)) / 5 if labels is not None else None
        return {'loss': loss, 'digit_logits': dl, 'fixation_info': info}

    def set_gumbel_temperature(self, t):
        for l in self.layers:
            if isinstance(l, SaccLayer): l.ctrl.temperature = t


# ── Train + Eval ──────────────────────────────────────────────────────────────

def gt(step):
    p = min(step / max(GUMBEL_ANNEAL_STEPS, 1), 1.0)
    return GUMBEL_TEMP_START + (GUMBEL_TEMP_END - GUMBEL_TEMP_START) * p

def train(model, device):
    ds = PasskeyDataset(NUM_TRAIN, TRAIN_CTX, seed=42)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    def lrf(s):
        if s < WARMUP_STEPS: return s / max(WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP_STEPS) / max(5000 - WARMUP_STEPS, 1)))
    sc = torch.optim.lr_scheduler.LambdaLR(opt, lrf)
    model.train()
    t0 = time.time()
    step = 0
    bud = WALL_CLOCK_BUDGET * 0.80
    while True:
        for b in ld:
            if time.time() - t0 >= bud:
                log(f'Budget: {step} steps in {time.time()-t0:.0f}s')
                return
            ids = b['input_ids'].to(device)
            dl = b['digit_labels'].to(device)
            model.set_gumbel_temperature(gt(step))
            out = model(ids, labels=dl)
            loss = out['loss']
            te = torch.tensor(0.0); ec = 0
            for _, inf in out['fixation_info'].items():
                for lg in inf['fixation_logits']:
                    p = F.softmax(lg, -1)
                    te = te + (-(p*(p+1e-8).log()).sum(-1).mean()).to(te.device); ec += 1
            if ec: loss = loss - ENTROPY_BONUS * te / ec
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw = torch.tensor(0.0, device=device); sc2 = 0
                tgt = torch.tensor([p // BLOCK_SIZE for p in b['passkey_position']], device=device, dtype=torch.long)
                for _, inf in out['fixation_info'].items():
                    for lg in inf['fixation_logits']:
                        sw = sw + F.cross_entropy(lg, tgt.clamp(max=lg.shape[1]-1)); sc2 += 1
                if sc2: loss = loss + w * sw / sc2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sc.step(); step += 1
            if step % 200 == 0:
                log(f'  step {step} | loss {loss.item():.4f} | {time.time()-t0:.0f}s')

def evaluate(model, ctx_len, device):
    ds = PasskeyDataset(NUM_EVAL, ctx_len, seed=99999)
    bs = max(1, min(16, 16384 // ctx_len * 4))
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    cor = tot = 0; dists = []
    with torch.no_grad():
        for b in ld:
            ids = b['input_ids'].to(device)
            out = model(ids)
            for _, inf in out['fixation_info'].items():
                for fp in inf['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(abs(fp[i].item() - b['passkey_position'][i]))
            for i in range(ids.shape[0]):
                pred = ''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred == b['passkey'][i]: cor += 1
                tot += 1
    return cor / max(tot, 1), sum(dists) / max(len(dists), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    # Train once at 2048
    max_pos = max(EVAL_CONTEXTS)  # position embeddings large enough for longest eval
    model = Model(max_pos).to(device)
    log(f'Parameters: {model._np:,}')
    log(f'Training at context={TRAIN_CTX} for {WALL_CLOCK_BUDGET}s...')
    train(model, device)

    # Evaluate at each context length WITHOUT retraining
    results = []
    for ctx in EVAL_CONTEXTS:
        log(f'Evaluating at context={ctx}...')
        acc, dist = evaluate(model, ctx, device)
        log(f'  context={ctx}: accuracy={acc:.4f}, distance={dist:.2f}')
        results.append({'context_length': ctx, 'accuracy': acc, 'fixation_distance': dist})
        torch.cuda.empty_cache()

    # Print results
    print('context_length\tpasskey_accuracy\tfixation_distance')
    for r in results:
        print(f'{r["context_length"]}\t{r["accuracy"]:.4f}\t{r["fixation_distance"]:.2f}')

    with open('context_scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
