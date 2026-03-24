"""RULER-style task generators for saccadic attention evaluation.

7 tasks across 4 categories:
  1. Retrieval: S-NIAH, MK-NIAH, MV-NIAH
  2. Multi-hop: Variable Tracking (2/3/4-hop)
  3. Aggregation: Common Words Extraction
  4. QA: (skipped — requires external dataset)

Each task produces (input_text, labels, metadata) for training and evaluation.
"""

import random
import string

# ── Paul Graham-style filler text ─────────────────────────────────────────────

PG_ESSAYS = [
    "The most important thing in a startup is not the idea, but the people. Great founders can make almost any idea work, while mediocre founders will struggle even with the best idea. This is counterintuitive because ideas seem like the scarce resource, but in practice execution matters far more. The reason is that ideas change as you work on them, and the quality of that change depends on the quality of the people doing the work.",
    "When you start a company, the first thing you notice is how few things you know. You thought you understood the market, but actually you were wrong about almost everything. The good news is that everyone is wrong about almost everything at the start. The winners are the ones who figure out the truth fastest. Speed of learning is the most important competitive advantage in a startup.",
    "The best essays are like conversations with a smart friend. They take you through a chain of reasoning, where each step follows naturally from the last. The writer doesn't just state conclusions; they show you the path that led to those conclusions. This is harder than it sounds, because in real thinking the path is rarely straight. You have to clean it up while keeping the feeling of discovery.",
    "There are two types of schedule: the maker's schedule and the manager's schedule. Managers' days are divided into one-hour blocks. They can schedule meetings whenever they want because meetings are the norm. But makers need long stretches of uninterrupted time to do creative work. A single meeting in the middle of the afternoon can blow a whole half-day for a maker, because it breaks the flow they need.",
    "The hardest part of starting a company is the emotional roller coaster. One day you feel like you're building the next Google, and the next day you feel like the whole thing is falling apart. Both feelings are probably wrong. The reality is somewhere in between, but the swings between extremes are exhausting. The founders who succeed are the ones who can keep going through the downs.",
    "Good design is simple. Not simple in the sense of easy, but simple in the sense of having nothing unnecessary. This principle applies to everything from visual design to software architecture to company strategy. The hard part is knowing what's unnecessary, because it often takes a deep understanding of the problem to see what can be removed.",
    "Technology progresses through a series of replacements. Each new technology displaces the previous one, not by being better at everything, but by being better at the things that matter most to the marginal user. This means that established technologies often look superior by most metrics, but the new technology wins anyway because it's better on the one metric that matters.",
    "The best way to predict the future is to build it. This sounds like a platitude, but it's actually a useful strategy. Instead of trying to guess what the future will look like and then building for it, you can just start building things and see which ones work. The future is shaped by the things that get built, so builders have disproportionate influence over it.",
    "Writing clearly is thinking clearly. If you can't explain something in simple terms, you probably don't understand it well enough. This is why writing is such a useful tool for learning: it forces you to organize your thoughts and identify the gaps in your understanding. The act of trying to explain something clearly often leads to new insights.",
    "The biggest risk in a startup is not building something nobody wants. It's building something that some people want but not enough people want enough. The difference between a successful product and a failed one is often not whether it works, but whether it solves a problem that enough people care about strongly enough to switch from whatever they're currently using.",
]


def _make_filler(tokenizer, ctx_len, reserved_tokens):
    """Generate filler text from PG essays to fill context."""
    filler_texts = []
    total = 0
    budget = ctx_len - reserved_tokens
    rng = random.Random()
    while total < budget + 500:  # overshoot then truncate
        essay = rng.choice(PG_ESSAYS)
        tokens = tokenizer.encode(essay)
        filler_texts.extend(tokens)
        total += len(tokens)
    return filler_texts[:budget]


# ── Task 1a: Single Needle in a Haystack (S-NIAH) ────────────────────────────

CITIES = ['Chicago', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 'Mumbai',
          'Toronto', 'Seoul', 'Dubai', 'Madrid', 'Rome', 'Bangkok', 'Cairo', 'Lima']


class SingleNeedleTask:
    """Insert one key-value pair (city → 7-digit number) in essay filler."""
    NAME = 'S-NIAH'
    N_DIGITS = 7

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            city = rng2.choice(CITIES)
            number = ''.join(rng2.choices(string.digits, k=self.N_DIGITS))
            needle = f" The special magic {city} number is: {number}."
            query = f" What is the special magic {city} number? The number is:"

            needle_ids = self.tokenizer.encode(needle)
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + number)
            reserved = len(needle_ids) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved)
            pos = rng2.randint(0, len(filler))

            full = filler[:pos] + needle_ids + filler[pos:] + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in number], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': number,
                'needle_position': pos,
                'target_positions': [pos],
            })
        return samples


# ── Task 1b: Multi-Key Needle (MK-NIAH) ──────────────────────────────────────

class MultiKeyNeedleTask:
    """1 target needle + 3 distractor needles. Query asks for target city."""
    NAME = 'MK-NIAH'
    N_DIGITS = 7
    N_DISTRACTORS = 3

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            cities = rng2.sample(CITIES, 1 + self.N_DISTRACTORS)
            target_city = cities[0]
            numbers = [''.join(rng2.choices(string.digits, k=self.N_DIGITS))
                        for _ in range(1 + self.N_DISTRACTORS)]

            needles = [f" The special magic {c} number is: {n}."
                       for c, n in zip(cities, numbers)]
            needle_ids_list = [self.tokenizer.encode(n) for n in needles]
            query = f" What is the special magic {target_city} number? The number is:"
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + numbers[0])
            reserved = sum(len(n) for n in needle_ids_list) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved)
            # Insert needles at random positions
            positions = sorted(rng2.sample(range(len(filler)), min(len(cities), len(filler))))
            target_pos = positions[0]
            full = list(filler)
            for j in range(len(cities) - 1, -1, -1):
                p = positions[j] if j < len(positions) else rng2.randint(0, len(full))
                full = full[:p] + needle_ids_list[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in numbers[0]], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': numbers[0],
                'needle_position': target_pos,
                'target_positions': [target_pos],
            })
        return samples


# ── Task 1c: Multi-Value Needle (MV-NIAH) ────────────────────────────────────

class MultiValueNeedleTask:
    """4 needles with same key (city), different values. Must find ALL."""
    NAME = 'MV-NIAH'
    N_DIGITS = 7
    N_VALUES = 4

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            city = rng2.choice(CITIES)
            numbers = [''.join(rng2.choices(string.digits, k=self.N_DIGITS))
                        for _ in range(self.N_VALUES)]
            needles = [f" The special magic {city} number is: {n}." for n in numbers]
            needle_ids_list = [self.tokenizer.encode(n) for n in needles]
            query = f" What are all the special magic {city} numbers? The numbers are:"
            query_ids = self.tokenizer.encode(query)
            # Answer: all numbers separated by comma
            answer_text = " " + ", ".join(numbers)
            answer_ids = self.tokenizer.encode(answer_text)
            reserved = sum(len(n) for n in needle_ids_list) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved)
            positions = sorted(rng2.sample(range(max(1, len(filler))),
                                           min(self.N_VALUES, max(1, len(filler)))))
            full = list(filler)
            target_positions = []
            for j in range(self.N_VALUES - 1, -1, -1):
                p = positions[j] if j < len(positions) else rng2.randint(0, len(full))
                target_positions.insert(0, p)
                full = full[:p] + needle_ids_list[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            # Labels: (N_VALUES, N_DIGITS)
            labels = torch.tensor([[int(d) for d in n] for n in numbers], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': numbers,
                'needle_position': target_positions[0],
                'target_positions': target_positions,
            })
        return samples


# ── Task 2a: Variable Tracking ───────────────────────────────────────────────

class VariableTrackingTask:
    """Chain of variable bindings: X=Y, Y=Z, Z=W. Query: what is X?"""
    NAME = 'VT'
    N_DIGITS = 4  # shorter values for variable tracking

    def __init__(self, tokenizer, chain_length=3):
        self.tokenizer = tokenizer
        self.chain_length = chain_length

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        var_names = [f'VAR_{chr(65+i)}' for i in range(26)]  # VAR_A, VAR_B, ...

        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            # Pick chain variables
            chain_vars = rng2.sample(var_names, self.chain_length + 1)
            final_value = ''.join(rng2.choices(string.digits, k=self.N_DIGITS))

            # Build chain statements
            statements = []
            for j in range(self.chain_length):
                if j == self.chain_length - 1:
                    # Last link: assign the actual value
                    stmt = f" {chain_vars[j]} equals {final_value}."
                else:
                    # Intermediate link: assign to next variable
                    stmt = f" {chain_vars[j]} equals {chain_vars[j+1]}."
                statements.append(stmt)

            stmt_ids = [self.tokenizer.encode(s) for s in statements]
            query = f" What is the value of {chain_vars[0]}? The value is:"
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + final_value)
            reserved = sum(len(s) for s in stmt_ids) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved)
            seg = max(1, len(filler) // (self.chain_length + 1))
            full = list(filler)
            target_positions = []
            for j in range(self.chain_length - 1, -1, -1):
                p = min(j * seg + rng2.randint(0, seg), len(full))
                target_positions.insert(0, p)
                full = full[:p] + stmt_ids[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in final_value], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': final_value,
                'needle_position': target_positions[0],
                'target_positions': target_positions,
            })
        return samples


# ── Task 3a: Common Words Extraction ─────────────────────────────────────────

class CommonWordsTask:
    """Fill context with words from a Zipf distribution. Query top-10 most common."""
    NAME = 'CWE'
    TOP_K = 5  # predict top 5 words

    # Fixed word vocabulary (common English words)
    VOCAB = [
        'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world', 'life',
        'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
        'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother',
        'area', 'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye',
        'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service',
        'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
        'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'body', 'back',
        'parent', 'face', 'other', 'level', 'office', 'door', 'health', 'person', 'art',
        'war', 'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
        'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education', 'foot', 'boy',
        'age', 'policy', 'process', 'music', 'market', 'sense', 'product', 'effect', 'class',
    ]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.word_to_idx = {w: i for i, w in enumerate(self.VOCAB)}

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        import numpy as np

        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            np_rng = np.random.RandomState(rng2.randint(0, 2**31))

            # Generate Zipf-distributed word counts
            n_words = ctx_len // 6  # ~6 chars per word average
            # Draw from Zipf with alpha=1.5
            weights = np.array([1.0 / (i + 1)**1.5 for i in range(len(self.VOCAB))])
            weights /= weights.sum()
            word_indices = np_rng.choice(len(self.VOCAB), size=n_words, p=weights)

            # Build text
            words = [self.VOCAB[idx] for idx in word_indices]
            text = ' '.join(words)

            # Find top-K most frequent
            from collections import Counter
            counts = Counter(word_indices)
            top_k_indices = [idx for idx, _ in counts.most_common(self.TOP_K)]

            query = " What are the most common words in the text above? The most common words are:"
            text_ids = self.tokenizer.encode(text)
            query_ids = self.tokenizer.encode(query)

            full = text_ids[:ctx_len - len(query_ids)] + query_ids
            if len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            # Multi-hot label over vocab
            labels = torch.zeros(len(self.VOCAB), dtype=torch.float)
            for idx in top_k_indices:
                labels[idx] = 1.0

            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': [self.VOCAB[idx] for idx in top_k_indices],
                'needle_position': 0,
                'target_positions': [],  # no specific positions for aggregation
            })
        return samples


# ── Task Registry ─────────────────────────────────────────────────────────────

def get_task(task_name, tokenizer, **kwargs):
    """Get a task generator by name."""
    tasks = {
        'S-NIAH': lambda: SingleNeedleTask(tokenizer),
        'MK-NIAH': lambda: MultiKeyNeedleTask(tokenizer),
        'MV-NIAH': lambda: MultiValueNeedleTask(tokenizer),
        'VT-2': lambda: VariableTrackingTask(tokenizer, chain_length=2),
        'VT-3': lambda: VariableTrackingTask(tokenizer, chain_length=3),
        'VT-4': lambda: VariableTrackingTask(tokenizer, chain_length=4),
        'CWE': lambda: CommonWordsTask(tokenizer),
    }
    return tasks[task_name]()


def get_head_config(task_name):
    """Return (head_class, head_kwargs, num_saccades) for a task."""
    from saccadic_qwen import (
        DigitClassificationHead, MultiValueHead,
        VariableTrackingHead, WordPredictionHead,
    )
    configs = {
        'S-NIAH':  (DigitClassificationHead, {'input_dim': 1536, 'n_digits': 7}, 3),
        'MK-NIAH': (DigitClassificationHead, {'input_dim': 1536, 'n_digits': 7}, 3),
        'MV-NIAH': (MultiValueHead, {'input_dim': 1536, 'n_values': 4, 'n_digits': 7}, 5),
        'VT-2':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 2),
        'VT-3':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 3),
        'VT-4':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 4),
        'CWE':     (WordPredictionHead, {'input_dim': 1536, 'vocab_size': 100, 'k': 5}, 10),
    }
    return configs[task_name]
