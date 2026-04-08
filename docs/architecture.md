# Saccadic Attention: Biologically-Inspired Foveated Attention for Efficient Long-Context Transformers

## Project Overview

Build a novel attention mechanism for transformer language models inspired by the human visual system's saccadic eye movement and foveated processing. Instead of attending to all tokens uniformly (standard attention) or using static sparse patterns, this model learns a **sequential fixation policy** that dynamically decides where in the context to focus high-resolution attention, while maintaining low-resolution peripheral awareness of the full context.

**The core insight:** The human visual system solves the same computational problem that long-context LLMs face — processing a massive high-dimensional input under strict compute/energy constraints. Biology's solution: foveated resolution (high acuity at fixation point, rapidly degrading periphery) combined with saccadic eye movements (a learned policy that moves the fixation point ~3 times per second to gather information sequentially). We apply this same strategy to transformer attention over token sequences.

**This is NOT a reproduction.** The closest prior work — the Fovea Transformer (He et al., arXiv:2311.07102) — uses a static fine-to-coarse attention pattern where nearby tokens get full attention and distant tokens are pooled into coarser representations. It is foveated but NOT saccadic: the pattern is the same for every query, there is no learned fixation policy, and there is no iterative sequential processing. Our key novelty is:
1. A **learned saccadic controller** that dynamically decides where to fixate based on peripheral context
2. **Iterative multi-fixation processing** — the model accumulates understanding across sequential fixation steps
3. **Modality-agnostic design** — this is an information-gathering strategy, not a vision-specific trick

## Architecture

The system has three components that work together in an iterative loop:

### Component 1: Peripheral Encoder (O(n) global awareness)

A lightweight module that processes the FULL context at low resolution to produce a "peripheral map" — a compressed representation of the entire sequence that tells the saccadic controller roughly what's where.

**Implementation:** Pool consecutive tokens into blocks and summarize each block:
```
Input: token embeddings x ∈ R^(n × d)
1. Divide x into blocks of size B: blocks ∈ R^(n/B × B × d)
2. Pool each block: peripheral_map ∈ R^(n/B × d)
   - Use learned weighted average (not just mean): a small Linear(d, 1) produces weights per token in each block, softmax, then weighted sum
3. Add learned positional embeddings to peripheral_map
```
Cost: O(n) — just a linear scan with a tiny network per block.

### Component 2: Foveal Processor (O(k²) local attention)

Standard multi-head self-attention, but ONLY over a small window of k tokens centered at the current fixation point. This is where the deep, expensive processing happens — but it's cheap because k << n.

**Implementation:**
```
Input: fixation_point (integer position), full token embeddings x, current state
1. Extract window: w = x[fixation_point - k//2 : fixation_point + k//2]  (k tokens)
2. Inject state: prepend the current accumulated state as a special token to w
3. Run standard multi-head self-attention over this (k+1)-length sequence
4. The output corresponding to the state token becomes the updated state
5. Optionally: apply a resolution gradient within the window itself
   (tokens closest to fixation center get full attention, tokens at window edges get attenuated)
```
Cost: O(k²) per fixation. With k=128 on a 32K context, this is ~16K ops vs ~1B for full attention.

### Component 3: Saccadic Controller (learned fixation policy)

A small network that takes the peripheral map and current accumulated state, and outputs a probability distribution over fixation positions. This is the novel piece — it learns WHERE to look next.

**Implementation:**
```
Input: peripheral_map ∈ R^(n/B × d), current_state ∈ R^d, fixation_history
1. Query: project current_state into a query vector q = Linear(current_state)
2. Score each peripheral block: scores = peripheral_map @ q / sqrt(d)
3. Mask previously fixated regions (optional — prevents re-fixation)
4. Apply temperature-scaled softmax to get fixation distribution
5. During training: sample using Gumbel-softmax (straight-through estimator) for differentiability
   During inference: take argmax
6. Convert block index to token position: fixation_point = selected_block * B
```
Cost: O(n/B) — a single matmul over the compressed peripheral map.

### The Forward Pass (iterative saccadic loop)

```python
def saccadic_attention(x, num_saccades=5, window_size=128, block_size=32):
    # Phase 1: Peripheral encoding (one-time, O(n))
    peripheral_map = peripheral_encoder(x)
    
    # Phase 2: Iterative saccadic fixation
    state = initial_state  # learned parameter or pooled from peripheral_map
    fixation_history = []
    
    for t in range(num_saccades):
        # Decide where to look
        fixation_point = saccadic_controller(peripheral_map, state, fixation_history)
        fixation_history.append(fixation_point)
        
        # Process at high resolution
        state = foveal_processor(x, fixation_point, state, window_size)
    
    # Phase 3: Final output
    return output_projection(state)
```

**Total cost:** O(n) + O(S × k²) where S = num_saccades, k = window_size.
For S=5, k=128, n=32768: ~82K ops vs ~1.07B for full attention. That's ~13,000x cheaper.

## Integration with GPT-2

We are NOT pretraining from scratch. We start with a pretrained GPT-2 (124M parameters) and replace the attention mechanism in selected layers with our saccadic attention module.

**Strategy: surgical replacement with frozen backbone**

```
GPT-2 Architecture (12 layers):
- Layers 0-5:  Keep original attention (FROZEN) — these handle local patterns
- Layers 6-11: Replace with saccadic attention (TRAINABLE) — these handle long-range dependencies
```

The GPT-2 token embeddings, FFN layers, and layer norms stay frozen. Only the new saccadic attention parameters (peripheral encoder, saccadic controller, foveal processor weights) are trained. This keeps the experiment lightweight and isolates the effect of our attention mechanism.

**Alternative simpler approach (start here):** Replace attention in ALL layers but keep everything except the saccadic components frozen. Fine-tune on a long-context task.

## Evaluation Plan

### Primary benchmark: Passkey Retrieval

The simplest and most telling test for long-context attention. Bury a random passkey (e.g., "The passkey is 48291") at a random position in a long document of filler text. The model must retrieve the passkey when asked at the end. This directly tests whether the saccadic controller can learn to find and fixate on the relevant information.

- Context lengths: 2K, 4K, 8K, 16K, 32K
- Metric: exact match accuracy
- Baseline: GPT-2 with full attention (which will fail beyond ~1K), GPT-2 with sliding window attention

### Secondary benchmark: PG19 perplexity

Evaluate language modeling perplexity on PG19 (long books). Compare:
- Full attention GPT-2 (truncated to 1024 tokens)
- Sliding window attention (window = 512)
- Our saccadic attention (5 saccades, window = 128, full context)
- Fovea-style static pattern (ablation — our architecture but without the learned controller)

### Key ablation studies

1. **Number of saccades:** 1, 3, 5, 10 — how many fixations does the model need?
2. **Window size:** 64, 128, 256, 512 — how big should the fovea be?
3. **Static vs. learned fixation:** Replace the saccadic controller with random fixation points or evenly-spaced fixation points. Does LEARNING where to look matter?
4. **Fixation visualization:** Plot where the model chooses to fixate across different inputs. Does it learn to fixate on semantically important regions (named entities, key facts, structural markers)?

The fixation visualization is the MONEY FIGURE for the paper. If we can show that the model learns to "read" a document by jumping between key information regions — the way human eyes saccade between salient features in a scene — that's the result that makes this publishable.

## Project Structure

```
saccadic-attention/
├── README.md                    # Project overview, results, figures
├── requirements.txt
├── CLAUDE.md                    # Claude Code project context
├── src/
│   ├── __init__.py
│   ├── peripheral_encoder.py    # Component 1: lightweight global context
│   ├── foveal_processor.py      # Component 2: high-res local attention
│   ├── saccadic_controller.py   # Component 3: learned fixation policy
│   ├── saccadic_attention.py    # Full module combining all three
│   ├── gpt2_saccadic.py         # GPT-2 with saccadic attention integrated
│   ├── data.py                  # Data loading, passkey generation
│   └── utils.py                 # Visualization, metrics, helpers
├── train.py                     # Training script
├── evaluate.py                  # Evaluation and benchmarking
├── notebooks/
│   ├── 01_architecture_demo.ipynb    # Visualize the attention mechanism
│   ├── 02_passkey_results.ipynb      # Passkey retrieval results
│   └── 03_fixation_analysis.ipynb    # Where does the model look? (money figure)
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   └── ablations/               # Configs for ablation studies
└── tests/
    ├── test_peripheral.py
    ├── test_foveal.py
    ├── test_controller.py
    └── test_integration.py
```

## Implementation Order (follow this exactly)

### Phase 1: Foundation (get the pieces working independently)

**Step 1:** Set up the project structure, requirements.txt, and CLAUDE.md.

Requirements:
```
torch>=2.0
transformers>=4.35
datasets
einops
matplotlib
pyyaml
tqdm
wandb  # optional, for experiment tracking
```

**Step 2:** Implement `peripheral_encoder.py`
- Input: (batch, seq_len, hidden_dim) tensor
- Output: (batch, num_blocks, hidden_dim) tensor
- Use learned weighted pooling within each block
- Add positional embeddings to the block representations
- Write test: verify output shapes, verify gradient flow

**Step 3:** Implement `foveal_processor.py`
- Input: full token embeddings, fixation point (integer), current state vector, window size
- Output: updated state vector
- Extract window around fixation point (handle edge cases at sequence boundaries)
- Use standard multi-head attention from PyTorch (nn.MultiheadAttention) over the window
- Prepend state as a CLS-like token, extract its output as updated state
- Write test: verify output shapes, verify it works at sequence boundaries

**Step 4:** Implement `saccadic_controller.py`
- Input: peripheral map, current state, optional fixation history
- Output: fixation point (integer position), fixation logits (for visualization)
- Cross-attention score between state query and peripheral map keys
- Gumbel-softmax for differentiable discrete selection during training
- Argmax during inference
- Write test: verify it produces valid positions, verify Gumbel-softmax gradient flow

**Step 5:** Implement `saccadic_attention.py` — the combined module
- Wire all three components together in the iterative loop
- Implement as a nn.Module with a clean forward() method
- Make num_saccades, window_size, block_size configurable
- Write integration test: random input → verify output shape matches standard attention output shape

### Phase 2: GPT-2 Integration

**Step 6:** Implement `gpt2_saccadic.py`
- Load pretrained GPT-2 from HuggingFace transformers
- Replace the attention mechanism in specified layers with SaccadicAttention
- Freeze all original GPT-2 parameters
- Only the new saccadic components should be trainable
- The foveal processor should reuse GPT-2's pretrained Q/K/V projection weights where possible
- Verify: model.parameters() should show correct trainable vs frozen counts

**Step 7:** Implement `data.py`
- Passkey retrieval dataset generator:
  - Generate documents of filler text (random Wikipedia-style or lorem ipsum)
  - Insert a passkey ("The secret number is XXXXX") at a random position
  - Append a retrieval prompt at the end ("What is the secret number?")
  - Return (input_ids, passkey_position, correct_answer)
- PG19 data loader (use HuggingFace datasets)

### Phase 3: Training and Evaluation

**Step 8:** Implement `train.py`
- Training loop for fine-tuning saccadic GPT-2 on passkey retrieval
- Use AdamW with cosine learning rate schedule
- Log: loss, passkey accuracy, fixation positions per step
- Checkpoint saving

**Step 9:** Implement `evaluate.py`
- Evaluate on passkey retrieval at multiple context lengths
- Evaluate PG19 perplexity
- Run ablation studies
- Generate fixation visualizations

**Step 10:** Create analysis notebooks
- `01_architecture_demo.ipynb`: Show the three components, visualize attention patterns
- `02_passkey_results.ipynb`: Accuracy vs context length, comparison to baselines
- `03_fixation_analysis.ipynb`: THE MONEY FIGURE — heatmaps of where the model fixates across different inputs. Does it learn to find the passkey? Does it learn to fixate on semantically important tokens in natural text?

### Phase 4: Polish

**Step 11:** Write README.md with:
- One-paragraph description of the idea
- Architecture diagram (ASCII art or a saved figure)
- Key results table
- Fixation visualization figure
- Installation and usage instructions
- Citation of relevant prior work (Fovea Transformer, Native Sparse Attention, etc.)
- Honest discussion of limitations and future work

## Key Design Decisions and Gotchas

1. **Gumbel-softmax temperature:** Start with τ=1.0 and anneal to 0.1 over training. Too high = too uniform fixations. Too low = gradient vanishing.

2. **Initial state:** Initialize as the mean of the peripheral map (gives the model a "first impression" of the whole context before any saccades).

3. **Foveal processor weight initialization:** Where possible, copy the pretrained GPT-2 attention weights into the foveal processor. This gives it a huge head start — it already knows how to attend within a window.

4. **Fixation diversity:** Add an entropy bonus to the saccadic controller loss to encourage diverse fixation positions (prevent the model from always fixating on the same position).

5. **Handling variable-length sequences:** Pad peripheral maps and mask appropriately. The saccadic controller should not select padded positions.

6. **Gradient flow through discrete selection:** The Gumbel-softmax straight-through estimator is the standard approach. Alternative: REINFORCE with baseline, but this is higher variance and harder to tune.

7. **Causal masking:** For autoregressive language modeling, the foveal processor should only attend to tokens at or before the current generation position. The saccadic controller should only fixate on past context, not future tokens.

## Metrics to Report

- Passkey retrieval accuracy at {2K, 4K, 8K, 16K, 32K} context lengths
- PG19 perplexity comparison
- FLOPs per forward pass vs full attention (theoretical + measured)
- Wall-clock inference time vs full attention
- Fixation entropy (how spread out are the fixations?)
- Fixation-to-target distance on passkey task (does it learn to find the key?)

## Related Work to Cite

- Fovea Transformer (He et al., 2311.07102) — static fine-to-coarse, no learned policy
- Native Sparse Attention (Yuan et al., 2502.11089) — DeepSeek, hardware-aligned sparse
- Sparse Frontier (Nawrot et al., 2504.17768) — Meta, sparse attention trade-offs survey
- MoBA (Lu et al., 2502.13189) — Mixture of Block Attention
- Learning When Not to Attend Globally (Luo et al., 2512.22562) — closest to saccadic idea
- SeerAttention (Gao et al., 2410.13276) — learned sparse patterns
- System 2 Attention (Weston & Sukhbaatar, 2311.11829) — Meta, deliberate attention
- Human Eyes Inspired RNNs (Choi et al., 2206.07282) — saccadic CNNs for vision
- Saccade-inspired ViT (Dallain et al., 2603.09613) — March 2026, saccadic image classification
- Active inference / free energy principle (Friston) — theoretical foundation for saccadic policy as information-gathering

## Biological Motivation (for README and eventual paper)

The human retina has ~6 million cone cells concentrated in the fovea (central ~2° of visual field), providing high-acuity color vision. Outside the fovea, resolution drops rapidly — peripheral vision is ~10-20x lower resolution. Despite this severe bottleneck, humans achieve remarkable visual understanding by executing ~3 saccadic eye movements per second, each guided by peripheral saliency signals and top-down task demands. 

This is not just a biological curiosity — it's an **optimal information-gathering strategy under compute constraints**. The saccadic system maximizes expected information gain per fixation, effectively performing active inference. We apply this same principle to transformer attention: instead of spending quadratic compute to attend uniformly over long contexts, we learn a policy that directs high-resolution processing to the most informative regions, building understanding iteratively across multiple fixation steps.

The connection is deeper than analogy. Standard transformer attention is analogous to a camera sensor that processes every pixel at full resolution simultaneously. Our saccadic attention is analogous to the biological visual system: foveated resolution with a learned scanpath. The information-theoretic optimality of saccadic strategies (maximizing mutual information between fixations and task-relevant variables) provides a principled foundation for the architecture, not just biological inspiration.
