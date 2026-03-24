"""Overnight RULER experiment: Saccadic Qwen vs Baseline Qwen at multiple context lengths.

For each task × context length:
  1. Train saccadic adapter until convergence
  2. Evaluate saccadic model (200 test samples)
  3. Evaluate baseline Qwen (same 200 test samples)
  4. Save results to ruler_results.json after each task completes

Convergence: val accuracy no improvement for 1000 steps, or 100%, or 30 min cap.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from saccadic_qwen import SaccadicQwen
from ruler_tasks import get_task, get_head_config

# ── Config ────────────────────────────────────────────────────────────────────

TASKS = ['S-NIAH', 'MK-NIAH', 'VT-2', 'VT-3', 'VT-4']
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]  # start at 2048 to validate pipeline
NUM_TRAIN = 2000
NUM_VAL = 100
NUM_EVAL = 200
BATCH_SIZE = 2
LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
SUPERVISED_WARMUP_STEPS = 500
SUPERVISED_WARMUP_WEIGHT = 3.0
TEMP_ANNEAL_STEPS = 800  # soft routing temperature annealing
MAX_TIME = 1800  # 30 min per config
PATIENCE = 2000  # match debug script — model needs ~600 steps before first improvement
VAL_EVERY = 50
ENTROPY_BONUS = 0.01
BLOCK_SIZE = 8

RESULTS_FILE = 'ruler_results.json'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B'            # base model for saccadic (frozen backbone)
INSTRUCT_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'  # instruct model for baseline eval

def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate(batch):
    result = {'input_ids': torch.stack([b['input_ids'] for b in batch])}
    # Labels can vary by task — stack if same shape, else handle per-task
    if batch[0]['labels'].dim() == 1:
        result['labels'] = torch.stack([b['labels'] for b in batch])
    elif batch[0]['labels'].dim() == 2:
        result['labels'] = torch.stack([b['labels'] for b in batch])
    else:
        result['labels'] = torch.stack([b['labels'] for b in batch])
    result['target_positions'] = [b.get('target_positions', []) for b in batch]
    result['answer'] = [b.get('answer', '') for b in batch]
    return result


# ── Gumbel temperature ───────────────────────────────────────────────────────

def routing_temp(step):
    """Anneal from 5.0 (soft) to 0.1 (hard) over training."""
    p = min(step / max(TEMP_ANNEAL_STEPS, 1), 1.0)
    return 5.0 + (0.1 - 5.0) * p


# ── Quick validation ─────────────────────────────────────────────────────────

def quick_val_accuracy(model, val_loader, task_name, device):
    """Compute accuracy on validation set."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            out = model(ids)
            if 'digit_logits' in out:
                for i in range(ids.shape[0]):
                    pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                    ans = batch['answer'][i]
                    if isinstance(ans, list):
                        ans = ans[0]  # for VT
                    if pred == str(ans):
                        correct += 1
                    total += 1
            elif 'all_logits' in out:
                # MV-NIAH: check if ANY value is correct
                for i in range(ids.shape[0]):
                    all_preds = []
                    for v_logits in out['all_logits']:
                        pred = ''.join(str(dl[i].argmax().item()) for dl in v_logits)
                        all_preds.append(pred)
                    answers = batch['answer'][i]
                    # Check if predicted set matches answer set
                    if set(all_preds) == set(answers):
                        correct += 1
                    total += 1
            elif 'word_logits' in out:
                # CWE: check top-K accuracy
                for i in range(ids.shape[0]):
                    pred_top = out['word_logits'][i].topk(5).indices.tolist()
                    ans = batch['answer'][i]
                    # Import task vocab to convert
                    from ruler_tasks import CommonWordsTask
                    pred_words = set(CommonWordsTask.VOCAB[j] for j in pred_top if j < len(CommonWordsTask.VOCAB))
                    ans_words = set(ans) if isinstance(ans, list) else {ans}
                    overlap = len(pred_words & ans_words) / max(len(ans_words), 1)
                    correct += overlap
                    total += 1
    model.train()
    return correct / max(total, 1)


# ── Supervised warmup loss ────────────────────────────────────────────────────

def supervised_fixation_loss(fixation_info, target_positions, block_size, device):
    """Push fixation logits toward known target positions."""
    total = torch.tensor(0.0, device=device)
    count = 0
    # Flatten all target positions per batch element
    batch_size = len(target_positions)
    for _, info in fixation_info.items():
        for s_idx, logits in enumerate(info['fixation_logits']):
            n_blocks = logits.shape[1]
            targets = []
            for b in range(batch_size):
                tps = target_positions[b]
                if tps:
                    # Cycle through target positions across saccades
                    tp = tps[s_idx % len(tps)]
                    targets.append(min(tp // block_size, n_blocks - 1))
                else:
                    targets.append(0)  # no target — dummy
            tgt = torch.tensor(targets, device=device, dtype=torch.long)
            total = total + F.cross_entropy(logits, tgt)
            count += 1
    return total / max(count, 1)


# ── Training until convergence ────────────────────────────────────────────────

def train_saccadic(model, train_data, val_data, task_name, device):
    """Train until convergence. Returns (steps, time, best_val_acc, reason)."""
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    def lr_fn(s):
        if s < WARMUP_STEPS: return s / max(WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP_STEPS) / 5000))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    model.train()
    t0 = time.time()
    step = 0
    best_val = 0.0
    since_improvement = 0
    reason = 'max_time'

    while True:
        for batch in train_loader:
            elapsed = time.time() - t0
            if elapsed >= MAX_TIME:
                reason = f'max_time ({MAX_TIME}s)'
                log(f'  MAX TIME: step {step}, best_val={best_val:.4f}')
                return step, elapsed, best_val, reason

            # Validation check
            if step > 0 and step % VAL_EVERY == 0:
                val_acc = quick_val_accuracy(model, val_loader, task_name, device)
                if val_acc > best_val:
                    best_val = val_acc
                    since_improvement = 0
                    log(f'  step {step} | val={val_acc:.4f} (NEW BEST) | {elapsed:.0f}s')
                    if val_acc >= 1.0:
                        reason = 'perfect'
                        return step, elapsed, best_val, reason
                else:
                    since_improvement += VAL_EVERY
                    if step % 200 == 0:
                        log(f'  step {step} | val={val_acc:.4f} (best={best_val:.4f}, plat={since_improvement}) | {elapsed:.0f}s')
                if since_improvement >= PATIENCE:
                    reason = f'plateau ({PATIENCE} steps)'
                    log(f'  PLATEAU: step {step}, best_val={best_val:.4f}')
                    return step, elapsed, best_val, reason

            # Forward
            ids = batch['input_ids'].to(device)
            labs = batch['labels'].to(device)
            model.set_gumbel_temperature(routing_temp(step))

            out = model(ids, labels=labs)
            loss = out['loss']

            # Entropy bonus
            ent_total = torch.tensor(0.0, device=device)
            ent_count = 0
            for _, info in out['fixation_info'].items():
                for logits in info['fixation_logits']:
                    p = F.softmax(logits, dim=-1)
                    ent_total = ent_total + (-(p * (p + 1e-8).log()).sum(-1).mean())
                    ent_count += 1
            if ent_count:
                loss = loss - ENTROPY_BONUS * ent_total / ent_count

            # Supervised warmup
            if SUPERVISED_WARMUP_STEPS > 0 and step < SUPERVISED_WARMUP_STEPS:
                w = SUPERVISED_WARMUP_WEIGHT * (1 - step / SUPERVISED_WARMUP_STEPS)
                sw = supervised_fixation_loss(
                    out['fixation_info'], batch['target_positions'], BLOCK_SIZE, device)
                loss = loss + w * sw

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()
            sched.step()
            step += 1


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_saccadic(model, test_data, task_name, device):
    """Evaluate saccadic model. Returns (accuracy, avg_fixation_distance)."""
    loader = DataLoader(test_data, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate)
    model.eval()
    correct = total = 0
    dists = []

    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            out = model(ids)

            # Fixation distances
            for _, info in out['fixation_info'].items():
                for fp in info['fixation_points']:
                    for i in range(ids.shape[0]):
                        tps = batch['target_positions'][i]
                        if tps:
                            min_dist = min(abs(fp[i].item() - tp) for tp in tps)
                            dists.append(min_dist)

            # Accuracy (task-specific)
            if 'digit_logits' in out:
                for i in range(ids.shape[0]):
                    pred = ''.join(str(dl[i].argmax().item()) for dl in out['digit_logits'])
                    ans = batch['answer'][i]
                    if isinstance(ans, list): ans = ans[0]
                    if pred == str(ans): correct += 1
                    total += 1
            elif 'all_logits' in out:
                for i in range(ids.shape[0]):
                    all_preds = []
                    for v_logits in out['all_logits']:
                        pred = ''.join(str(dl[i].argmax().item()) for dl in v_logits)
                        all_preds.append(pred)
                    if set(all_preds) == set(batch['answer'][i]): correct += 1
                    total += 1
            elif 'word_logits' in out:
                from ruler_tasks import CommonWordsTask
                for i in range(ids.shape[0]):
                    pred_top = out['word_logits'][i].topk(5).indices.tolist()
                    pred_words = set(CommonWordsTask.VOCAB[j] for j in pred_top if j < len(CommonWordsTask.VOCAB))
                    ans_words = set(batch['answer'][i])
                    correct += len(pred_words & ans_words) / max(len(ans_words), 1)
                    total += 1

    acc = correct / max(total, 1)
    dist = sum(dists) / max(len(dists), 1)
    return acc, dist


def evaluate_baseline(test_data, task_name, ctx_len, device, base_tokenizer):
    """Evaluate Qwen-Instruct with chat template. Substring match for accuracy."""
    instruct_tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL, dtype=torch.float16, trust_remote_code=True).to(device)
    qwen.eval()

    correct = total = 0
    n_eval = min(50, len(test_data))  # cap baseline eval (generation is slow)
    with torch.no_grad():
        for sample in test_data[:n_eval]:
            # Decode the raw input_ids back to text (using base tokenizer)
            raw_text = base_tokenizer.decode(sample['input_ids'].tolist(), skip_special_tokens=True)

            # Format as chat with explicit instruction
            if task_name == 'CWE':
                instruction = "List the 5 most common words in the text above, separated by commas. Reply with ONLY the words."
            elif 'VT' in task_name:
                instruction = "Follow the variable assignments in the text and answer the question. Reply with ONLY the final numeric value."
            elif 'MV' in task_name:
                instruction = "Find ALL matching numbers in the text and list them. Reply with ONLY the numbers separated by commas."
            else:
                instruction = "Answer the question at the end of the text. Reply with ONLY the number, nothing else."

            messages = [{"role": "user", "content": raw_text + "\n\n" + instruction}]
            prompt = instruct_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                ids = instruct_tok.encode(prompt, return_tensors="pt").to(device)
                # Skip if too long for GPU memory
                if ids.shape[1] > ctx_len + 500:
                    ids = ids[:, -(ctx_len + 500):]  # keep end (where query is)
                gen = qwen.generate(ids, max_new_tokens=50, do_sample=False,
                                    pad_token_id=instruct_tok.eos_token_id)
                gen_text = instruct_tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                log(f'  Baseline OOM at ctx_len={ctx_len}, skipping remaining')
                torch.cuda.empty_cache()
                break
            except Exception as e:
                gen_text = ''
                log(f'  Baseline error: {e}')

            ans = sample['answer']
            if isinstance(ans, list):
                if all(str(a) in gen_text for a in ans):
                    correct += 1
            else:
                if str(ans) in gen_text:
                    correct += 1
            total += 1

    del qwen
    torch.cuda.empty_cache()
    return correct / max(total, 1)


# ── Load/save results ────────────────────────────────────────────────────────

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def already_done(results, task_name, ctx_len):
    return any(r['task'] == task_name and r['context_length'] == ctx_len for r in results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = load_results()
    log(f'Loaded {len(results)} existing results')

    for task_name in TASKS:
        for ctx_len in CONTEXT_LENGTHS:
            if already_done(results, task_name, ctx_len):
                log(f'SKIP {task_name} @ {ctx_len} (already done)')
                continue

            log(f'\n{"="*60}')
            log(f'{task_name} @ context_length={ctx_len}')
            log(f'{"="*60}')

            # Generate data
            task = get_task(task_name, tokenizer)
            log(f'Generating {NUM_TRAIN} train + {NUM_VAL} val + {NUM_EVAL} test samples...')
            train_data = task.generate(ctx_len, NUM_TRAIN, seed=42)
            val_data = task.generate(ctx_len, NUM_VAL, seed=77777)
            test_data = task.generate(ctx_len, NUM_EVAL, seed=99999)

            # Get task head config
            head_cls, head_kwargs, num_saccades = get_head_config(task_name)

            # Build saccadic model
            log(f'Building SaccadicQwen (num_saccades={num_saccades})...')
            task_head = head_cls(**head_kwargs)
            model = SaccadicQwen(
                model_name=MODEL_NAME,
                saccadic_dim=128,
                block_size=BLOCK_SIZE,
                window_size=64,
                num_saccades=num_saccades,
                n_heads=4,
                task_head=task_head,
            ).to(device)
            log(f'Trainable params: {model.trainable_params():,}')

            # Train saccadic model
            log('Training saccadic model...')
            steps, elapsed, best_val, reason = train_saccadic(
                model, train_data, val_data, task_name, device)

            # Evaluate saccadic
            log('Evaluating saccadic model...')
            sacc_acc, sacc_dist = evaluate_saccadic(model, test_data, task_name, device)
            log(f'Saccadic: accuracy={sacc_acc:.4f}, distance={sacc_dist:.2f}')

            # Free saccadic model
            del model
            torch.cuda.empty_cache()

            # Evaluate baseline Qwen (Instruct model with chat template)
            log('Evaluating baseline Qwen-Instruct...')
            try:
                baseline_acc = evaluate_baseline(test_data, task_name, ctx_len, device, tokenizer)
                log(f'Baseline: accuracy={baseline_acc:.4f}')
            except Exception as e:
                log(f'Baseline failed: {e}')
                baseline_acc = -1  # mark as failed

            torch.cuda.empty_cache()

            # Save result
            result = {
                'task': task_name,
                'context_length': ctx_len,
                'saccadic_accuracy': sacc_acc,
                'saccadic_fixation_distance': sacc_dist,
                'baseline_accuracy': baseline_acc,
                'saccadic_steps': steps,
                'saccadic_time_s': round(elapsed, 1),
                'converge_reason': reason,
                'num_saccades': num_saccades,
            }
            results.append(result)
            save_results(results)
            log(f'SAVED: {task_name} @ {ctx_len}: sacc={sacc_acc:.4f}, base={baseline_acc:.4f}')

    # Print final summary
    log('\n' + '='*60)
    log('FINAL SUMMARY')
    log('='*60)
    print('task\tcontext_length\tsaccadic_accuracy\tbaseline_accuracy\tfixation_distance\tsteps\ttime_s')
    for r in results:
        print(f'{r["task"]}\t{r["context_length"]}\t{r["saccadic_accuracy"]:.4f}\t'
              f'{r["baseline_accuracy"]:.4f}\t{r["saccadic_fixation_distance"]:.2f}\t'
              f'{r["saccadic_steps"]}\t{r["saccadic_time_s"]}')


if __name__ == '__main__':
    main()
