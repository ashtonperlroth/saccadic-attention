# Saccadic Attention — Autonomous Experiment Program

You are an autonomous research agent optimizing a saccadic attention mechanism for GPT-2. You run experiments in a loop, keeping improvements and reverting failures. You never stop. The human may be asleep.

## Metric

**Primary:** passkey retrieval accuracy at 4096 context length (higher is better, range 0.0–1.0).
**Secondary:** average fixation-to-target distance (lower is better, measured in tokens).

An experiment is an **improvement** if passkey_accuracy strictly increases. If accuracy is tied, lower fixation_distance breaks the tie. All else being equal, simpler is better — a tiny gain from deleting code is a definite keep, a tiny gain that adds 20 lines of complexity is probably not worth it.

## Setup (one-time, before the loop)

1. `cd` to the project root (where this file lives).
2. Create and switch to a branch: `git checkout -b autoresearch/run-$(date +%s)`
3. Run the **baseline** experiment with the unmodified config:
   ```
   python experiment.py > run.log 2>&1
   ```
4. `grep "^passkey_accuracy:\|^fixation_distance:" run.log` to get the baseline numbers.
5. Record the baseline in `results.tsv` with status `baseline`.
6. `git add -A && git commit -m "baseline"`

## The Experiment Loop

**LOOP FOREVER:**

1. **Hypothesize.** Look at `results.tsv` and recent git log. Think about what to try next. Write a short hypothesis (1–2 sentences).

2. **Modify.** Make targeted changes. You may edit:
   - `configs/default.yaml` — hyperparameters
   - `src/peripheral_encoder.py` — peripheral encoder design
   - `src/foveal_processor.py` — foveal processor internals
   - `src/saccadic_controller.py` — controller architecture, selection mechanism
   - `src/saccadic_attention.py` — state initialization, output projection, loop structure
   - `src/gpt2_saccadic.py` — which layers get saccadic attention, weight initialization
   - `train.py` — optimizer, schedule, loss computation
   - `experiment.py` — only the training hyperparameters section, NOT the evaluation protocol

3. **Commit before running.**
   ```
   git add -A && git commit -m "<short description of what you changed>"
   ```

4. **Run the experiment.**
   ```
   python experiment.py > run.log 2>&1
   ```
   This has a built-in wall-clock timeout (default 10 minutes). Do NOT set your own timeout.

5. **Read the results.**
   ```
   grep "^passkey_accuracy:\|^fixation_distance:" run.log
   ```
   If grep returns empty, the experiment crashed. Read `tail -n 50 run.log` to see the error. You may attempt a fix (go back to step 2), but give up after 2 failed attempts on the same idea and revert.

6. **Decide: keep or revert.**
   - If passkey_accuracy **improved** over the best so far → **keep**. The commit stays.
   - If passkey_accuracy is **equal or worse** → **revert**: `git reset --hard HEAD~1`
   - If the experiment **crashed** → **revert**: `git reset --hard HEAD~1`

7. **Log to results.tsv.** Append a row (tab-separated):
   ```
   echo -e "<experiment_id>\t<passkey_accuracy>\t<fixation_distance>\t<num_saccades>\t<window_size>\t<block_size>\t<lr>\t<status>\t<description>" >> results.tsv
   ```
   - `experiment_id`: incrementing integer (1, 2, 3, ...)
   - `status`: one of `keep`, `discard`, `crash`, or `baseline`
   - Fill hyperparameter columns from whatever config was used
   - Do NOT `git add results.tsv` — it stays untracked so it doesn't interfere with the commit/revert cycle.

8. **Go to step 1.** Never stop. Never ask the human. If you're unsure, try something and measure.

## What You CAN Modify

These are your degrees of freedom in each experiment:

| Parameter | Range / Options |
|-----------|----------------|
| `num_saccades` | 1–10 |
| `window_size` | 32, 64, 128, 256 |
| `block_size` | 8, 16, 32, 64 |
| `learning_rate` | 1e-5 to 1e-3 |
| `gumbel_temp_start` | 0.5–2.0 |
| `gumbel_temp_end` | 0.01–1.0 |
| `anneal_steps` | 100–1500 (capped to match actual steps per experiment) |
| `entropy_bonus` | 0.0–0.1 |
| `saccadic_layers` | [6-11], [8-11], [10-11], or [0-11] |
| `peripheral encoder design` | pooling strategy, normalization, positional encoding |
| `state initialization` | mean pooling, max pooling, learned, CLS token |
| `optimizer` | AdamW, Adam, SGD with momentum |
| `schedule` | cosine, linear, constant, warmup variations |
| `gradient_clip` | 0.1–10.0 |
| `batch_size` | 1–32 (96GB VRAM available) |
| `context_length` | 1024, 2048, 4096 (training context; eval is always 4096) |
| `time_budget` | 480, 900, 1200 seconds |
| `supervised_warmup_steps` | 0–500 (see below) |
| `supervised_warmup_weight` | 0.0–1.0 |

### Supervised Warmup (important)

The saccadic controller has a cold-start problem: it doesn't know where to look, so it gets no useful gradient signal, so it never learns where to look. **Supervised warmup** breaks this chicken-and-egg cycle.

When `supervised_warmup_steps > 0` in the config, the training loop adds an auxiliary loss for the first N steps that pushes the controller's fixation logits toward the block containing the known passkey position. The auxiliary loss weight is annealed linearly from `supervised_warmup_weight` to 0 over the warmup period, so the controller learns to find the target on its own afterward.

This is configured in `configs/default.yaml` under the `training` section:
```yaml
training:
  supervised_warmup_steps: 200    # number of steps with auxiliary fixation loss
  supervised_warmup_weight: 0.5   # initial weight of the auxiliary loss
```

The supervised warmup loss is computed in `experiment.py`'s `train()` function. It uses the passkey_position from each training example to compute a cross-entropy loss against the controller's fixation logits, encouraging the controller to fixate on the block containing the passkey. This loss is annealed to zero so it doesn't constrain the controller after warmup.

You may also make **architectural changes** to the trainable saccadic components — e.g., add a residual connection in the controller, change the foveal processor's FFN, modify how peripheral blocks are pooled. Be creative.

## What You CANNOT Modify

These are invariants. Violating any of these makes the experiment invalid.

1. **Base model**: Always GPT-2 124M (`gpt2` from HuggingFace). Always frozen.
2. **Three-component architecture**: The system must always have a peripheral encoder, foveal processor, and saccadic controller. You cannot merge them, remove one, or bypass the saccadic loop.
3. **Core forward pass structure**: `peripheral_map → for t in range(num_saccades): controller selects fixation → foveal processor updates state → output`. The iterative saccadic loop must remain.
4. **Evaluation protocol**: The eval section of `experiment.py` (passkey dataset generation, accuracy computation, distance computation) must not be modified. Results must be comparable across experiments.
5. **Output format**: `experiment.py` must always print exactly `passkey_accuracy: X.XX` and `fixation_distance: X.XX` to stdout.

## Strategy Suggestions

These are starting points, not mandates. Use your judgment.

- **Start with hyperparameter sweeps** before architectural changes. Low-hanging fruit: learning rate, num_saccades, window_size.
- **The controller is the novel piece.** Improvements to how fixation points are selected will likely have the biggest impact.
- **Entropy bonus matters.** Too low → the model always fixates on the same spot. Too high → fixations are random. Find the sweet spot.
- **Layer selection matters.** Maybe saccadic attention helps more in later layers. Maybe earlier layers benefit too. Try different subsets.
- **Watch for mode collapse.** If fixation_distance is very high and not decreasing, the controller may not be learning. Try higher entropy bonus or different initialization.
- **Gumbel temperature annealing is critical.** Too fast → gradient issues. Too slow → training is noisy. The anneal schedule is a key hyperparameter.
- **Think about what a human reader would do.** If you were scanning a long document for a hidden number, what strategy would you use? Encode that intuition into the architecture.

## File Layout Reference

```
saccadic-attention/
├── program.md              ← you are here (read-only)
├── experiment.py           ← self-contained train+eval script
├── results.tsv             ← experiment log (untracked)
├── run.log                 ← latest experiment output (untracked)
├── configs/default.yaml    ← hyperparameters (editable)
├── src/
│   ├── peripheral_encoder.py   (editable)
│   ├── foveal_processor.py     (editable)
│   ├── saccadic_controller.py  (editable)
│   ├── saccadic_attention.py   (editable)
│   ├── gpt2_saccadic.py        (editable)
│   ├── data.py                 (editable)
│   └── utils.py                (editable)
├── train.py                    (editable)
└── evaluate.py                 (editable)
```
