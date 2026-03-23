# Saccadic Attention: Architecture Ideas & Research Directions

> This document captures every idea discussed during the development of the saccadic attention architecture. Ideas are organized from "implement next" to "future work / paper ideas." Each idea includes the biological motivation, the technical implementation sketch, and what result it would demonstrate.

---

## Current Architecture (v1 — what's running now)

Three components in a loop:

1. **Peripheral encoder**: Raw token embeddings → averaged into blocks of 32 → 128 blurry block vectors. Static. Runs once.
2. **Saccadic controller**: Cross-attention between state vector and peripheral map → Gumbel-softmax → picks one block to fixate on.
3. **Foveal processor**: Full multi-head attention over a 128-token window centered at fixation. Updates a single state vector.

The state vector is the only thing that carries information between saccades. The peripheral map never changes. Each saccade is independent except through the state bottleneck.

**Current results (as of experiment 5):** 42.5% passkey accuracy at 4096 context length, fixation distance 190. Controller is learning to find the passkey. Still optimizing via autoresearch loop.

---

## v2 Ideas: Implement After Single-Hop Plateaus

### Idea 1: Accumulated Foveal Context (Working Memory)

**The problem:** Currently, each saccade sees ONLY its own 128-token window + a single state vector summarizing all previous saccades. The state vector is a lossy bottleneck — details from earlier fixations fade.

**The idea:** Instead of compressing previous fixations into a state vector, KEEP the actual tokens from all previous fixation windows. Each saccade's attention window grows:

```
Saccade 1: [128 tokens]                              = 128 tokens
Saccade 2: [128 from saccade 1] + [128 new]          = 256 tokens
Saccade 3: [128 from s1] + [128 from s2] + [128 new] = 384 tokens
Saccade 5: all five windows concatenated              = 640 tokens
```

The foveal processor runs full self-attention over the growing window. The model can directly cross-reference tokens from different fixation regions — no information loss.

**Biological motivation:** Human working memory holds ~4 items (Cowan, 2001). Each "item" is a fixation's worth of information. You don't compress your last fixation into a single number — you hold the actual visual details in working memory and can compare them to what you're currently looking at.

**Cost:** Quadratic in the number of saccades × window size. Saccade k costs (k × 128)². For 5 saccades: total ≈ 128² + 256² + 384² + 512² + 640² ≈ 819K operations. Still 20x cheaper than full attention on 4096 tokens (16.7M).

**What it enables:** Multi-hop reasoning. The model can find clue 1 in saccade 1, find clue 2 in saccade 2, and then directly attend between the tokens of both clues to compose the answer. Impossible with the current state-vector bottleneck.

**Variant — foveated memory decay:** Previous fixation windows get progressively lower resolution the older they are. Current fixation: full resolution. Previous fixation: pooled 2x (64 effective tokens). Two fixations ago: pooled 4x (32 effective tokens). This keeps the window growth slower while still retaining more than a single state vector. More biologically accurate — your memory of 3 saccades ago is fuzzier than 1 saccade ago.

---

### Idea 2: Global Peripheral Map Update (Belief Revision)

**The problem:** The peripheral map is computed once from raw token embeddings and never changes. The saccadic controller makes all its decisions based on this stale, uninformed map. After saccade 1 reveals "this is a legal contract," the controller still sees the same blurry blocks it saw before — it doesn't know to look for "financial terms" or "party names."

**The idea:** After each saccade, the ENTIRE peripheral map transforms based on what the fovea found. Not just the blocks that were fixated — every block in the map updates, because learning something about one part of the document changes what you'd expect to find everywhere else.

**Implementation:** Cross-attention where all 128 peripheral blocks (queries) attend to the foveal output (keys/values):

```python
for t in range(num_saccades):
    fixation = controller(peripheral_map, state)
    foveal_output, state = foveal_processor(tokens, fixation, state)
    
    # Global belief update — the whole map reacts to what the fovea found
    peripheral_map = cross_attention(
        query=peripheral_map,       # 128 blocks: "how does this change me?"
        key=foveal_output,          # what the fovea learned
        value=foveal_output
    ) + peripheral_map              # residual connection
```

Cost: 128 × 128 = 16K operations per saccade. Trivial.

**Biological motivation:** Top-down modulation in visual cortex. Higher cortical areas (which have processed foveal input) send signals BACK to lower areas, changing how they represent unfixated regions. Your peripheral vision isn't passively reporting raw pixels — it's actively shaped by what you've already recognized. This is the core mechanism in predictive coding theory.

**What it enables:** Semantically-informed saccade planning. After learning "this is a legal contract" from saccade 1, the controller doesn't have to randomly search for relevant information — the updated peripheral map now highlights blocks that are likely to contain parties, dates, financial terms, etc. Each saccade becomes more targeted because the map itself has become smarter.

**Important safeguard — confidence-weighted updates:** Early saccades should influence the map less than later saccades (the model is less certain after 1 fixation than after 4). Implementation: `alpha = sigmoid(linear(saccade_step_embedding))`, scaling the update strength. Prevents the first (possibly atypical) fixation from hijacking the entire map.

**Important safeguard — residual blending:** Don't replace the peripheral map, blend: `map = (1 - alpha) * old_map + alpha * updated_map`. No single saccade can revolutionize the map; changes accumulate gradually across multiple saccades.

---

### Idea 3: Richer Peripheral Encoding

**The problem:** The current peripheral encoder averages 32 raw token embeddings per block. This destroys almost all useful structure. A block containing "The secret number is 48291" averages to roughly the same mushy vector as a block containing "The quick brown fox jumps over" — both are just the centroid of a bunch of word embeddings.

Human peripheral vision extracts much more: scene gist, spatial layout, ensemble statistics (variance, texture, dominant features), and rough semantic categories — all before any saccade.

**Sub-idea 3a: Build from layer-5 outputs, not raw embeddings.**

GPT-2 layers 0-5 run standard full attention (only layers 6-11 are saccadic in our design). By layer 5, token representations are contextualized — they encode local syntax, entity types, short-range relationships. Building the peripheral map from layer-5 outputs instead of raw embeddings is essentially free (that computation happens anyway) and makes every block dramatically more informative.

This is probably a one-line change and should be the FIRST thing we implement in v2.

**Sub-idea 3b: Multiple statistics per block, not just mean.**

```python
block_summary = project(concat(
    mean(block_tokens),      # average content
    std(block_tokens),       # how varied is this region? (filler = low, passkey = high)
    max_pool(block_tokens),  # most "extreme" token (unusual patterns stand out)
))
```

The standard deviation feature is especially powerful — blocks of repetitive filler have low variance, while blocks containing unusual content (numbers among words, a passkey, a name) have high variance. The controller can learn "saccade toward high-variance blocks" as a general heuristic, even before understanding what's in them. This is exactly what human peripheral vision does — you notice "something different over there" before knowing what it is.

**Sub-idea 3c: Tiny intra-block attention.**

Instead of pooling, run a single self-attention head within each 32-token block. The model learns which tokens within each block are most important and weights them accordingly. Cost: 32² × 128 blocks = 131K operations. Still negligible.

---

## v3 Ideas: For Multi-Hop Experiments & Paper

### Idea 4: Multi-Hop Passkey Task

**The experiment that tests whether saccadic attention enables compositional reasoning.**

Instead of one passkey, place N clues at random positions:
```
Clue 1 (position ~800):  "The first digit is 7"
Clue 2 (position ~2400): "The second digit is 3"
Clue 3 (position ~3600): "The third digit is 9"
Question: "What is the 3-digit code?"
Answer: 739
```

Test with N = 2, 3, 4, 5 hops. For each N, evaluate accuracy vs. number of saccades. 

**The key prediction:** N-hop tasks should require approximately N saccades. If accuracy on 3-hop tasks improves when going from 3 to 5 saccades, but 1-hop tasks don't benefit from extra saccades, that demonstrates the model is using additional saccades for additional reasoning steps — not just redundant search.

**Why this is important:** This would show that saccadic attention is not just efficient retrieval but a new form of iterative reasoning. Standard full attention (fixed depth) provably cannot compose more than a constant number of hops. Saccadic attention with N saccades can compose N hops. That's a strictly larger computational class.

**The accumulated foveal context (Idea 1) is probably necessary for this to work.** With the current state-vector bottleneck, the model has to compress "first digit is 7" into a lossy vector before finding the second clue. With accumulated context, it can directly cross-reference all found clues at full resolution.

---

### Idea 5: Saccade Scaling Laws

**The experiment that tests whether saccades are a new axis of intelligence scaling.**

Current scaling laws describe three axes: parameters, data, and test-time compute (chain-of-thought depth). If saccadic attention works for multi-hop reasoning, there's a fourth axis: test-time spatial exploration (number of saccades).

**The experiment:** For fixed model size and fixed training, plot accuracy vs. num_saccades for tasks of increasing difficulty. If you see consistent scaling — more saccades = better performance on harder tasks — that's a scaling law. Plot it on a log-log scale. If it's a power law (straight line on log-log), that's a publishable finding.

**Critical control:** Also plot accuracy vs. total FLOPs (not just num_saccades) to separate "more saccades help because more compute" from "more saccades help because more sequential reasoning steps." If a model with 10 saccades × 128 window beats a model with 2 saccades × 512 window at the same total FLOPs, the benefit is from sequential reasoning, not just more compute. That's the result that proves saccades are a genuinely new scaling axis.

---

### Idea 6: Unfreezing the LM Head / Last Layers

**The observation:** In our experiments, fixation distance dropped much faster than accuracy improved. The controller learned to FIND the passkey quickly, but the model struggled to EXTRACT and REPRODUCE it. This suggests the bottleneck shifted from the controller to the frozen GPT-2 layers.

**The experiment:** Progressively unfreeze:
1. Just the LM head (final projection to vocabulary)
2. LM head + last 2 transformer layers
3. LM head + last 4 layers

If unfreezing the LM head causes a big accuracy jump with minimal distance change, it confirms the bottleneck was extraction, not localization. This would also mean our controller is better than the accuracy numbers suggest — it's finding the passkey but the frozen GPT-2 can't act on it.

---

### Idea 7: Multi-Scale Peripheral Hierarchy

**The deepest version of the peripheral encoding idea.**

Instead of one map at one resolution, build a hierarchy:

```
Level 0: 4096 individual token representations (full detail, only accessed by foveal processor)
Level 1: 128 block summaries (32 tokens each — current peripheral map)  
Level 2: 16 section summaries (256 tokens each — very coarse)
Level 3: 1 document summary (the whole thing — "gist")
```

The saccadic controller consults ALL levels:
- Level 3: "this is a legal document" → filters which Level 2 sections are relevant
- Level 2: "section 7 has financial terms" → filters which Level 1 blocks to examine  
- Level 1: "block 94 has unusual statistics" → precise fixation target

After each saccade, the update cascades both UP (foveal → blocks → sections → gist) and DOWN (updated gist → updated section expectations → updated block expectations).

**Biological motivation:** This exactly mirrors the visual cortex hierarchy — V1 (fine detail) ↔ V2 (textures) ↔ V4 (shapes) ↔ IT (objects) ↔ PFC (scene understanding), with both feedforward and feedback connections at every level. Recurrent processing between levels is what makes human visual recognition robust.

**Implementation complexity:** High. Save for v3 or future work section of paper.

---

### Idea 8: Active Inference Objective

**The idea:** Instead of training the saccadic controller purely on task loss (passkey accuracy), add an information-theoretic objective: the controller should maximize expected information gain per saccade.

**Formally:** The controller should select fixation points that maximize the mutual information between the foveal observation and the task-relevant variables (the answer). This is equivalent to minimizing the expected entropy of the answer distribution after observing the fixated region.

**Connection to Karl Friston's free energy principle:** The saccadic system is performing active inference — selecting actions (fixation points) to minimize surprise (prediction error / free energy). The peripheral map encodes the model's current generative model of the document. Unfixated regions have high uncertainty (high entropy). The controller fixates where uncertainty is highest AND where resolution is most likely to be informative for the task.

**Practical implementation:** Add an auxiliary loss that encourages the fixation distribution to have high entropy (explore diverse positions) early in training, transitioning to an information-gain objective later. The information gain can be approximated as the KL divergence between the answer distribution before and after a fixation.

**This reframes the entire architecture in the language of Bayesian optimal experimental design** — each saccade is an experiment, the controller is choosing experiments to maximally reduce posterior uncertainty about the answer.

---

## Theoretical Connections (for paper framing)

### Connection 1: Breaking the TC⁰ Ceiling

Standard fixed-depth transformers are provably limited to the circuit class TC⁰ (Merrill & Sabharwal). They cannot solve problems requiring computational depth that scales with input length — like multi-hop composition, graph connectivity, or Boolean formula evaluation.

Saccadic attention introduces a variable-depth iterative loop. Each saccade adds effective computational depth. With N saccades, the model can perform N sequential reasoning steps conditioned on previous results. This is mathematically equivalent to chain-of-thought (which Merrill & Sabharwal proved breaks the TC⁰ ceiling), but instead of generating intermediate tokens, the model reasons by choosing where to direct attention.

**Key claim:** Saccadic attention with unbounded saccades can simulate any polynomial-time computation, placing it in the complexity class P — strictly larger than TC⁰.

### Connection 2: Test-Time Compute as Spatial Exploration

The current test-time compute paradigm (o1, o3, DeepSeek-R1) scales along one dimension: depth (more reasoning tokens). Saccadic attention adds a second dimension: spatial exploration (more fixation points across the context). 

The combination of both — think longer AND look at more evidence — is an unexplored region of the test-time compute landscape. A unified framework that adaptively allocates BOTH depth and spatial breadth based on task difficulty would be a genuine theoretical contribution.

### Connection 3: Predictive Coding

The global peripheral map update (Idea 2) implements predictive coding: the peripheral map is the generative model, the foveal processor computes prediction errors, and the global update propagates those errors back into the model. The saccadic controller selects the next observation to maximize prediction error reduction (active inference).

This maps the architecture directly onto the leading computational theory of the visual cortex (Rao & Ballard, 1999; Friston, 2005).

### Connection 4: Global Workspace Theory

The accumulated foveal context (Idea 1) + globally-updated peripheral map (Idea 2) together implement something like Baars' Global Workspace Theory of consciousness. The working memory (accumulated fixation windows) is the "global workspace" — a shared representational space that integrates information from multiple perceptual acts. The peripheral map is the "unconscious background" that contextualizes but doesn't directly enter the workspace.

This is a stretch for a paper (invoking consciousness is risky), but it's a legitimate intellectual connection if framed carefully.

---

## Experiment Priority Queue

Once the autoresearch loop plateaus on single-hop:

1. **Build peripheral map from layer-5 outputs** (Idea 3a) — one-line change, should immediately improve controller quality
2. **Add variance/max-pool features to peripheral blocks** (Idea 3b) — small change, helps controller distinguish content types
3. **Implement accumulated foveal context** (Idea 1) — prerequisite for multi-hop
4. **Implement global peripheral map update** (Idea 2) — makes saccade planning semantically informed
5. **Design and run multi-hop passkey experiments** (Idea 4) — the key result for the paper
6. **Run saccade scaling experiments** (Idea 5) — the "new scaling axis" result
7. **Unfreeze LM head** (Idea 6) — if accuracy is bottlenecked on extraction
8. **Information-theoretic controller objective** (Idea 8) — principled training, harder to implement
9. **Multi-scale peripheral hierarchy** (Idea 7) — complex, save for v3

---

## Key Figures for Paper

1. **Fixation heatmaps:** Where does the model look across different inputs? Does it learn interpretable scanpaths?
2. **Accuracy vs. context length:** Fixed saccades/window, scaling context from 1K to 64K. Does accuracy hold? (constant-compute attention)
3. **Accuracy vs. num_saccades for N-hop tasks:** The scaling law figure. More saccades = more hops?
4. **Peripheral map evolution:** Visualize how the peripheral map changes across saccades (Idea 2). Does it become more semantically structured?
5. **FLOPs comparison:** Saccadic attention vs. full attention vs. sparse attention baselines, accuracy vs. compute.
6. **Ablation table:** With/without accumulated context, with/without peripheral update, with/without supervised warmup, static vs. learned controller.
