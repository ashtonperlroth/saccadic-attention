"""Self-contained experiment script: train saccadic GPT-2 on passkey retrieval and evaluate.

Prints exactly two lines to stdout:
    passkey_accuracy: X.XX
    fixation_distance: X.XX

All other output goes to stderr so the agent can grep stdout cleanly.
"""

import math
import sys
import time

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.data import PasskeyRetrievalDataset
from src.gpt2_saccadic import GPT2Saccadic


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG_PATH = 'configs/default.yaml'
WALL_CLOCK_BUDGET_SECONDS = 20 * 60  # 10 minutes total (train + eval)
EVAL_CONTEXT_LENGTH = 4096
TRAIN_CONTEXT_LENGTH = 4096
NUM_TRAIN_SAMPLES = 4000
NUM_EVAL_SAMPLES = 200
EVAL_BATCH_SIZE = 4


def log(msg: str):
    """Print to stderr so it doesn't pollute the two-line stdout output."""
    print(msg, file=sys.stderr, flush=True)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Training ───────────────────────────────────────────────────────────────────

def get_gumbel_temperature(step: int, config: dict) -> float:
    t_start = config['gumbel']['temp_start']
    t_end = config['gumbel']['temp_end']
    anneal_steps = config['gumbel']['anneal_steps']
    progress = min(step / max(anneal_steps, 1), 1.0)
    return t_start + (t_end - t_start) * progress


def compute_entropy_bonus(fixation_info: dict) -> torch.Tensor:
    total_entropy = torch.tensor(0.0)
    count = 0
    for layer_idx, info in fixation_info.items():
        for logits in info['fixation_logits']:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            total_entropy = total_entropy + entropy.to(total_entropy.device)
            count += 1
    return total_entropy / max(count, 1)


def compute_supervised_warmup_loss(
    fixation_info: dict,
    passkey_positions: list[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Auxiliary loss pushing fixation logits toward the block containing the passkey.

    Uses cross-entropy between the controller's fixation logits and the target
    block index derived from the known passkey position. Applied across all
    saccadic layers and all saccade steps so the controller gets dense signal.
    """
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # Target block index for each batch element: (batch,)
    target_blocks = torch.tensor(
        [pos // block_size for pos in passkey_positions],
        device=device, dtype=torch.long,
    )

    for layer_idx, info in fixation_info.items():
        for logits in info['fixation_logits']:
            # logits: (batch, num_blocks)
            num_blocks = logits.shape[1]
            # Clamp target to valid range (passkey might be in last partial block)
            clamped_targets = target_blocks.clamp(max=num_blocks - 1)
            total_loss = total_loss + torch.nn.functional.cross_entropy(logits, clamped_targets)
            count += 1

    return total_loss / max(count, 1)


def collate_fn(batch: list[dict]) -> dict:
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
    }


def train(model, config, tokenizer, device, time_budget: float) -> None:
    """Train for a fixed wall-clock budget."""
    tc = config['training']
    sc = config['saccadic']

    dataset = PasskeyRetrievalDataset(
        num_samples=NUM_TRAIN_SAMPLES,
        context_length=TRAIN_CONTEXT_LENGTH,
        tokenizer=tokenizer,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=tc['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=tc['lr'], weight_decay=tc['weight_decay'])
    # Estimate max steps from time budget (rough: assume ~1s per step as upper bound)
    max_steps = tc.get('max_steps', 50000)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=tc['lr'] * 0.1)

    model.train()
    start_time = time.time()
    global_step = 0
    epoch = 0

    while True:
        epoch += 1
        for batch in loader:
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                log(f'Time budget reached after {global_step} steps, {elapsed:.0f}s')
                return

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Gumbel temperature annealing
            temp = get_gumbel_temperature(global_step, config)
            model.set_gumbel_temperature(temp)

            # Forward
            outputs = model(input_ids, labels=labels)
            lm_loss = outputs['loss']

            # Entropy bonus
            entropy = compute_entropy_bonus(outputs['fixation_info'])
            entropy_coeff = tc.get('entropy_bonus', 0.01)
            loss = lm_loss - entropy_coeff * entropy

            # Supervised warmup: auxiliary loss pushing fixations toward passkey
            warmup_steps = tc.get('supervised_warmup_steps', 0)
            warmup_weight_init = tc.get('supervised_warmup_weight', 0.5)
            warmup_loss_val = 0.0
            if warmup_steps > 0 and global_step < warmup_steps:
                # Linear anneal from warmup_weight_init to 0 over warmup_steps
                warmup_weight = warmup_weight_init * (1.0 - global_step / warmup_steps)
                warmup_loss = compute_supervised_warmup_loss(
                    outputs['fixation_info'],
                    batch['passkey_position'],
                    sc['block_size'],
                    device,
                )
                loss = loss + warmup_weight * warmup_loss
                warmup_loss_val = warmup_loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            grad_clip = tc.get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                warmup_str = f' | warmup_loss {warmup_loss_val:.3f}' if warmup_steps > 0 and global_step <= warmup_steps else ''
                log(f'  step {global_step} | loss {loss.item():.4f} | '
                    f'lm_loss {lm_loss.item():.4f} | entropy {entropy.item():.3f} | '
                    f'temp {temp:.3f} | elapsed {elapsed:.0f}s{warmup_str}')


# ── Evaluation (DO NOT MODIFY THIS SECTION) ───────────────────────────────────

def evaluate(model, tokenizer, device) -> tuple[float, float]:
    """Evaluate passkey retrieval accuracy and fixation-to-target distance.

    DO NOT MODIFY THIS FUNCTION. Results must be comparable across experiments.

    Returns:
        (passkey_accuracy, mean_fixation_distance)
    """
    dataset = PasskeyRetrievalDataset(
        num_samples=NUM_EVAL_SAMPLES,
        context_length=EVAL_CONTEXT_LENGTH,
        tokenizer=tokenizer,
        seed=99999,  # fixed eval seed, different from training
    )
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.eval()
    correct = 0
    total = 0
    fixation_distances = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            passkeys = batch['passkey']
            passkey_positions = batch['passkey_position']

            outputs = model(input_ids)
            logits = outputs['logits']

            # Collect fixation-to-target distances
            for layer_idx, info in outputs['fixation_info'].items():
                for fp in info['fixation_points']:
                    for i in range(input_ids.shape[0]):
                        dist = abs(fp[i].item() - passkey_positions[i])
                        fixation_distances.append(dist)

            # Check passkey accuracy
            for i in range(input_ids.shape[0]):
                labels = batch['labels'][i]
                answer_positions = (labels != -100).nonzero(as_tuple=True)[0]
                if len(answer_positions) == 0:
                    continue

                pred_ids = logits[i, answer_positions - 1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_ids).strip()

                if passkeys[i] in pred_text:
                    correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    mean_distance = sum(fixation_distances) / max(len(fixation_distances), 1)

    return accuracy, mean_distance


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    sc = config['saccadic']

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log(f'Device: {device}')

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])

    # Model
    log('Loading GPT-2 and building saccadic model...')
    model = GPT2Saccadic(
        model_name=config['model']['name'],
        saccadic_layers=config['model']['saccadic_layers'],
        num_saccades=sc['num_saccades'],
        window_size=sc['window_size'],
        block_size=sc['block_size'],
        gumbel_temperature=config['gumbel']['temp_start'],
        mask_fixated=sc['mask_fixated'],
    ).to(device)

    trainable = model.get_trainable_params()
    frozen = model.get_frozen_params()
    log(f'Trainable: {trainable:,} | Frozen: {frozen:,}')

    # Budget: reserve 20% for eval, 80% for training
    total_budget = WALL_CLOCK_BUDGET_SECONDS
    train_budget = total_budget * 0.80
    log(f'Training budget: {train_budget:.0f}s | Total budget: {total_budget:.0f}s')

    # Train
    log('Starting training...')
    train_start = time.time()
    train(model, config, tokenizer, device, time_budget=train_budget)
    train_elapsed = time.time() - train_start
    log(f'Training complete in {train_elapsed:.0f}s')

    # Evaluate
    log('Evaluating...')
    accuracy, mean_distance = evaluate(model, tokenizer, device)
    log(f'Evaluation complete.')

    # Print the two result lines to stdout (this is what the agent greps for)
    print(f'passkey_accuracy: {accuracy:.4f}')
    print(f'fixation_distance: {mean_distance:.2f}')


if __name__ == '__main__':
    main()
