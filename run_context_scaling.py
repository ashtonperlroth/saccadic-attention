"""Context length scaling experiment.

Trains a separate model at each context length with the EXACT same config
(4 layers, 128 dim, 3 saccades, window=64, block=8). 4-minute budget each.
Tests whether accuracy holds as context grows while compute stays fixed.
"""

import json
import sys
import time

# Import everything from train_tiny — we use the same model and training code
from train_tiny import (
    TinySaccadicTransformer, PasskeyDataset, collate,
    train, log, WALL_CLOCK_BUDGET,
    DataLoader,
)
import train_tiny
import torch

CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384]
NUM_EVAL = 200


def run_one(ctx_len, device):
    """Train and evaluate at a single context length."""
    log(f'\n{"="*60}')
    log(f'Context length: {ctx_len}')
    log(f'{"="*60}')

    # Set train context = eval context for this experiment
    train_tiny.TRAIN_CONTEXT_LENGTH = ctx_len
    train_tiny.MAX_POS = max(ctx_len, train_tiny.MAX_POS)

    # Build fresh model with enough position embeddings
    old_max = train_tiny.MAX_POS
    train_tiny.MAX_POS = ctx_len
    model = TinySaccadicTransformer().to(device)
    train_tiny.MAX_POS = old_max
    log(f'Parameters: {model._n_params:,}')

    # Train
    t0 = time.time()
    train(model, device)
    log(f'Training done in {time.time()-t0:.0f}s')

    # Evaluate at this context length
    ds = PasskeyDataset(NUM_EVAL, ctx_len, seed=99999)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    correct = total = 0
    dists = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            out = model(ids)
            for layer_idx, info in out['fixation_info'].items():
                for fp in info['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(abs(fp[i].item() - batch['passkey_position'][i]))
            for i in range(ids.shape[0]):
                pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                if pred == batch['passkey'][i]:
                    correct += 1
                total += 1

    acc = correct / max(total, 1)
    dist = sum(dists) / max(len(dists), 1)
    log(f'Result: accuracy={acc:.4f}, distance={dist:.2f}')
    return {'context_length': ctx_len, 'accuracy': acc, 'fixation_distance': dist}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    log(f'Running context scaling: {CONTEXT_LENGTHS}')

    results = []
    for ctx_len in CONTEXT_LENGTHS:
        r = run_one(ctx_len, device)
        results.append(r)

    # Print results table to stdout
    print('context_length\tpasskey_accuracy\tfixation_distance')
    for r in results:
        print(f'{r["context_length"]}\t{r["accuracy"]:.4f}\t{r["fixation_distance"]:.2f}')

    # Save JSON
    with open('context_scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
