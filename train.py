"""Training script for saccadic attention GPT-2."""

import argparse
import math
import os

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from src.data import PasskeyRetrievalDataset
from src.gpt2_saccadic import GPT2Saccadic


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_gumbel_temperature(step: int, config: dict) -> float:
    """Linear anneal from temp_start to temp_end over anneal_steps."""
    t_start = config['gumbel']['temp_start']
    t_end = config['gumbel']['temp_end']
    anneal_steps = config['gumbel']['anneal_steps']
    progress = min(step / anneal_steps, 1.0)
    return t_start + (t_end - t_start) * progress


def compute_entropy_bonus(fixation_info: dict) -> torch.Tensor:
    """Compute entropy of fixation distributions to encourage diversity."""
    total_entropy = torch.tensor(0.0)
    count = 0
    for layer_idx, info in fixation_info.items():
        for logits in info['fixation_logits']:
            # logits: (batch, num_blocks)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            total_entropy = total_entropy + entropy.to(total_entropy.device)
            count += 1
    return total_entropy / max(count, 1)


def evaluate_passkey_accuracy(
    model: GPT2Saccadic,
    eval_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
) -> float:
    """Evaluate exact-match passkey retrieval accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            passkeys = batch['passkey']

            outputs = model(input_ids)
            logits = outputs['logits']  # (batch, seq_len, vocab)

            # Get the model's prediction at the position where the answer should be
            # The answer tokens are at the end of the sequence
            for i in range(input_ids.shape[0]):
                # Find where labels are not -100 (answer region)
                labels = batch['labels'][i]
                answer_positions = (labels != -100).nonzero(as_tuple=True)[0]
                if len(answer_positions) == 0:
                    continue

                # Get predicted tokens at answer positions
                pred_ids = logits[i, answer_positions - 1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_ids).strip()

                if passkeys[i] in pred_text:
                    correct += 1
                total += 1

    model.train()
    return correct / max(total, 1)


def collate_fn(batch: list[dict]) -> dict:
    """Collate function that handles string fields."""
    result = {}
    result['input_ids'] = torch.stack([b['input_ids'] for b in batch])
    result['labels'] = torch.stack([b['labels'] for b in batch])
    result['passkey'] = [b['passkey'] for b in batch]
    result['passkey_position'] = [b['passkey_position'] for b in batch]
    return result


def main():
    parser = argparse.ArgumentParser(description='Train Saccadic Attention GPT-2')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    tc = config['training']
    sc = config['saccadic']
    dc = config['data']

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])

    # Model
    model = GPT2Saccadic(
        model_name=config['model']['name'],
        saccadic_layers=config['model']['saccadic_layers'],
        num_saccades=sc['num_saccades'],
        window_size=sc['window_size'],
        block_size=sc['block_size'],
        gumbel_temperature=config['gumbel']['temp_start'],
        mask_fixated=sc['mask_fixated'],
    ).to(device)

    print(f'Trainable params: {model.get_trainable_params():,}')
    print(f'Frozen params:    {model.get_frozen_params():,}')

    # Datasets
    train_dataset = PasskeyRetrievalDataset(
        num_samples=dc['num_train_samples'],
        context_length=dc['train_context_length'],
        tokenizer=tokenizer,
        seed=42,
    )
    eval_dataset = PasskeyRetrievalDataset(
        num_samples=dc['num_eval_samples'],
        context_length=dc['train_context_length'],
        tokenizer=tokenizer,
        seed=123,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=tc['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=tc['lr'], weight_decay=tc['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc['max_steps'], eta_min=tc['lr'] * 0.1)

    # Optional wandb logging
    use_wandb = config['logging'].get('use_wandb', False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config['logging']['project'],
            name=config['logging'].get('run_name'),
            config=config,
        )

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0

    print(f'Starting training for {tc["max_steps"]} steps...')

    while global_step < tc['max_steps']:
        for batch in train_loader:
            if global_step >= tc['max_steps']:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Update Gumbel temperature
            temp = get_gumbel_temperature(global_step, config)
            model.set_gumbel_temperature(temp)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            lm_loss = outputs['loss']

            # Entropy bonus for fixation diversity
            entropy = compute_entropy_bonus(outputs['fixation_info'])
            loss = lm_loss - tc['entropy_bonus'] * entropy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, tc['gradient_clip'])
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % tc['log_every'] == 0:
                avg_loss = running_loss / tc['log_every']
                lr = scheduler.get_last_lr()[0]
                msg = (
                    f'Step {global_step}/{tc["max_steps"]} | '
                    f'Loss: {avg_loss:.4f} | LM Loss: {lm_loss.item():.4f} | '
                    f'Entropy: {entropy.item():.4f} | Temp: {temp:.3f} | LR: {lr:.2e}'
                )
                print(msg)

                if use_wandb:
                    wandb.log({
                        'loss': avg_loss,
                        'lm_loss': lm_loss.item(),
                        'entropy_bonus': entropy.item(),
                        'gumbel_temperature': temp,
                        'learning_rate': lr,
                        'step': global_step,
                    })
                running_loss = 0.0

            # Evaluation
            if global_step % tc['eval_every'] == 0:
                accuracy = evaluate_passkey_accuracy(model, eval_loader, tokenizer, device)
                print(f'  Eval passkey accuracy: {accuracy:.4f}')
                if use_wandb:
                    wandb.log({'passkey_accuracy': accuracy, 'step': global_step})
                model.train()

            # Checkpointing
            if global_step % tc['checkpoint_every'] == 0:
                ckpt_path = os.path.join(args.output_dir, f'checkpoint_{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': {
                        k: v for k, v in model.state_dict().items()
                        if any(k.startswith(f'saccadic_blocks.{l}')
                               for l in model.saccadic_blocks.keys())
                    },
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, ckpt_path)
                print(f'  Saved checkpoint: {ckpt_path}')

    # Final save
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': {
            k: v for k, v in model.state_dict().items()
            if any(k.startswith(f'saccadic_blocks.{l}')
                   for l in model.saccadic_blocks.keys())
        },
        'config': config,
    }, final_path)
    print(f'Training complete. Final model saved to {final_path}')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
