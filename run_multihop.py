"""Multi-hop passkey experiment.

N clues scattered across the context, each providing one digit of a N-digit code.
Tests accuracy vs num_saccades for N=2,3,4,5 hops.
Key prediction: N-hop tasks need ~N saccades.
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

from train_tiny import (
    CHARS, CHAR_TO_ID, ID_TO_CHAR, PAD_ID, VOCAB_SIZE,
    HIDDEN_DIM, N_HEADS, BLOCK_SIZE, WINDOW_SIZE,
    GUMBEL_TEMP_START, GUMBEL_TEMP_END, GUMBEL_ANNEAL_STEPS,
    BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP, ENTROPY_BONUS,
    WARMUP_STEPS, SUPERVISED_WARMUP_STEPS, SUPERVISED_WARMUP_WEIGHT,
    FILLER_SENTENCES,
    PeripheralEncoder, SaccadicController, FovealProcessor,
    SaccadicAttentionLayer, StandardAttentionLayer,
    encode, decode, log, entropy_bonus,
)
import train_tiny

CONTEXT_LENGTH = 2048
WALL_CLOCK_BUDGET = 300  # 5 minutes per experiment
NUM_TRAIN = 5000
NUM_EVAL = 200


# ── Multi-Hop Dataset ────────────────────────────────────────────────────────

class MultiHopPasskeyDataset(Dataset):
    """N clues scattered across context, each providing one digit."""

    def __init__(self, n_samples, ctx_len, n_hops, seed=42):
        self.n = n_samples
        self.ctx_len = ctx_len
        self.n_hops = n_hops
        self.rng = random.Random(seed)
        self.filler_encoded = [encode(s) for s in FILLER_SENTENCES]
        ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
        self.ordinals = ordinals[:n_hops]
        self.prompt_ids = encode(f' what is the {n_hops}-digit code? the code is ')
        self.samples = [self._make(i) for i in range(n_samples)]

    def _make(self, idx):
        rng = random.Random(self.rng.randint(0, 2**32) + idx)
        digits = [rng.choice(string.digits) for _ in range(self.n_hops)]
        code = ''.join(digits)

        # Build clue strings
        clues = []
        for i, (d, ordinal) in enumerate(zip(digits, self.ordinals)):
            clues.append(encode(f' the {ordinal} digit of the code is {d}. '))

        # Build filler
        answer_ids = encode(code)
        clue_total = sum(len(c) for c in clues)
        reserved = clue_total + len(self.prompt_ids) + len(answer_ids)
        filler_budget = self.ctx_len - reserved

        filler = []
        while len(filler) < filler_budget:
            filler.extend(rng.choice(self.filler_encoded))
        filler = filler[:filler_budget]

        # Insert clues at random positions (sorted so indices don't shift)
        # Divide filler into n_hops+1 segments
        seg_len = len(filler) // (self.n_hops + 1)
        clue_positions = []
        pieces = []
        offset = 0
        for i in range(self.n_hops):
            # Insert clue i somewhere in segment i
            seg_start = i * seg_len
            seg_end = (i + 1) * seg_len
            insert_at = rng.randint(seg_start, seg_end)
            clue_positions.append(insert_at + sum(len(c) for c in clues[:i]))

        # Build full sequence by inserting clues into filler
        full = list(filler)
        for i in range(self.n_hops - 1, -1, -1):
            pos = min(i * seg_len + rng.randint(0, seg_len), len(full))
            clue_positions[i] = pos
            full = full[:pos] + clues[i] + full[pos:]

        full = full + self.prompt_ids + answer_ids

        if len(full) > self.ctx_len:
            full = full[:self.ctx_len]
        elif len(full) < self.ctx_len:
            full = full + [PAD_ID] * (self.ctx_len - len(full))

        input_ids = torch.tensor(full, dtype=torch.long)
        digit_labels = torch.tensor([int(d) for d in code], dtype=torch.long)

        return input_ids, digit_labels, code, clue_positions

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input_ids, digit_labels, code, clue_positions = self.samples[idx]
        return {
            'input_ids': input_ids,
            'digit_labels': digit_labels,
            'passkey': code,
            'passkey_position': clue_positions[0],  # first clue position for fixation distance
            'clue_positions': clue_positions,
        }


def collate_multihop(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'digit_labels': torch.stack([b['digit_labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
        'clue_positions': [b['clue_positions'] for b in batch],
    }


# ── Model (same architecture, configurable n_hops and n_saccades) ────────────

class MultiHopSaccadicTransformer(nn.Module):
    def __init__(self, n_hops, n_saccades):
        super().__init__()
        self.n_saccades = n_saccades
        self.n_hops = n_hops
        self.tok_emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, HIDDEN_DIM)
        self.drop = nn.Dropout(0.1)
        # 2 standard + 2 saccadic (same as winning config)
        self.layers = nn.ModuleList([
            StandardAttentionLayer(HIDDEN_DIM, N_HEADS),
            StandardAttentionLayer(HIDDEN_DIM, N_HEADS),
            SaccadicAttentionLayer(HIDDEN_DIM, N_HEADS),
            SaccadicAttentionLayer(HIDDEN_DIM, N_HEADS),
        ])
        self.ln_f = nn.LayerNorm(HIDDEN_DIM)
        self.digit_heads = nn.ModuleList([nn.Linear(HIDDEN_DIM, 10) for _ in range(n_hops)])
        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        periph_src = None
        all_info = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SaccadicAttentionLayer):
                # Override NUM_SACCADES for this experiment
                old_ns = train_tiny.NUM_SACCADES
                train_tiny.NUM_SACCADES = self.n_saccades
                x, info = layer(x, periph_src=periph_src)
                train_tiny.NUM_SACCADES = old_ns
                all_info[i] = info
            else:
                x = layer(x)
                if i == 1:  # layer before first saccadic
                    periph_src = x.detach()
        x = self.ln_f(x)
        last_hidden = x[:, -1, :]
        digit_logits = [head(last_hidden) for head in self.digit_heads]
        loss = None
        if labels is not None:
            loss = sum(F.cross_entropy(digit_logits[i], labels[:, i]) for i in range(self.n_hops)) / self.n_hops
        return {'loss': loss, 'digit_logits': digit_logits, 'fixation_info': all_info}

    def set_gumbel_temperature(self, temp):
        for layer in self.layers:
            if isinstance(layer, SaccadicAttentionLayer):
                layer.controller.temperature = temp


# ── Training ──────────────────────────────────────────────────────────────────

def get_gumbel_temp(step):
    progress = min(step / max(GUMBEL_ANNEAL_STEPS, 1), 1.0)
    return GUMBEL_TEMP_START + (GUMBEL_TEMP_END - GUMBEL_TEMP_START) * progress


def train_multihop(model, n_hops, n_saccades, device):
    ds = MultiHopPasskeyDataset(NUM_TRAIN, CONTEXT_LENGTH, n_hops, seed=42)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_multihop)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_fn(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / max(5000 - WARMUP_STEPS, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    model.train()
    t0 = time.time()
    step = 0
    budget = WALL_CLOCK_BUDGET * 0.80

    while True:
        for batch in loader:
            if time.time() - t0 >= budget:
                log(f'  Budget reached: {step} steps in {time.time()-t0:.0f}s')
                return
            ids = batch['input_ids'].to(device)
            dlabs = batch['digit_labels'].to(device)
            model.set_gumbel_temperature(get_gumbel_temp(step))

            out = model(ids, labels=dlabs)
            loss = out['loss']
            ent = entropy_bonus(out['fixation_info'])
            loss = loss - ENTROPY_BONUS * ent

            # Supervised warmup: push fixation toward ALL clue positions
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw_total = torch.tensor(0.0, device=device)
                sw_count = 0
                for layer_idx, info in out['fixation_info'].items():
                    for s_idx, logits in enumerate(info['fixation_logits']):
                        n_blocks = logits.shape[1]
                        # Target: the clue position for this saccade step (cycle through clues)
                        clue_idx = s_idx % n_hops
                        targets = torch.tensor(
                            [cp[clue_idx] // BLOCK_SIZE for cp in batch['clue_positions']],
                            device=device, dtype=torch.long
                        ).clamp(max=n_blocks - 1)
                        sw_total = sw_total + F.cross_entropy(logits, targets)
                        sw_count += 1
                loss = loss + w * sw_total / max(sw_count, 1)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            step += 1

            if step % 200 == 0:
                log(f'    step {step} | loss {loss.item():.4f} | {time.time()-t0:.0f}s')


def eval_multihop(model, n_hops, device):
    ds = MultiHopPasskeyDataset(NUM_EVAL, CONTEXT_LENGTH, n_hops, seed=99999)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_multihop)
    model.eval()
    correct = total = 0
    min_dists = []  # min distance from any fixation to any clue
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            out = model(ids)
            # Accuracy: all N digits correct
            for i in range(ids.shape[0]):
                pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                if pred == batch['passkey'][i]:
                    correct += 1
                total += 1
            # Fixation distance: average min distance from each clue to nearest fixation
            for i in range(ids.shape[0]):
                all_fps = []
                for info in out['fixation_info'].values():
                    for fp in info['fixation_points']:
                        all_fps.append(fp[i].item())
                for cp in batch['clue_positions'][i]:
                    if all_fps:
                        min_dists.append(min(abs(fp - cp) for fp in all_fps))

    acc = correct / max(total, 1)
    avg_dist = sum(min_dists) / max(len(min_dists), 1)
    return acc, avg_dist


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    results = []
    hop_counts = [2, 3, 4, 5]

    for n_hops in hop_counts:
        saccade_counts = [n_hops, n_hops + 1, n_hops + 2]
        for n_saccades in saccade_counts:
            log(f'\n{"="*60}')
            log(f'N_HOPS={n_hops}, N_SACCADES={n_saccades}')
            log(f'{"="*60}')

            model = MultiHopSaccadicTransformer(n_hops, n_saccades).to(device)
            log(f'Parameters: {model._n_params:,}')

            train_multihop(model, n_hops, n_saccades, device)
            acc, dist = eval_multihop(model, n_hops, device)

            log(f'Result: accuracy={acc:.4f}, avg_clue_distance={dist:.2f}')
            results.append({
                'n_hops': n_hops,
                'n_saccades': n_saccades,
                'accuracy': acc,
                'avg_clue_distance': dist,
            })

    # Print results table to stdout
    print('n_hops\tn_saccades\taccuracy\tavg_clue_distance')
    for r in results:
        print(f'{r["n_hops"]}\t{r["n_saccades"]}\t{r["accuracy"]:.4f}\t{r["avg_clue_distance"]:.2f}')

    # Save JSON
    with open('multihop_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
