"""Data loading: passkey retrieval dataset and PG19 loader."""

import random
import string

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


# Filler sentences for passkey retrieval task
FILLER_SENTENCES = [
    "The weather was pleasant and the sky was clear.",
    "Several researchers gathered to discuss the latest findings.",
    "The library contained thousands of books on various topics.",
    "Traffic moved slowly through the busy intersection.",
    "The conference room was filled with eager participants.",
    "A gentle breeze rustled through the autumn leaves.",
    "The project deadline was approaching rapidly.",
    "Students worked diligently on their assignments.",
    "The old building stood at the corner of the street.",
    "New developments in technology continued to emerge.",
    "The garden was well maintained throughout the year.",
    "Several factors contributed to the overall outcome.",
    "The meeting was scheduled for early in the morning.",
    "A small group discussed various approaches to the problem.",
    "The document outlined the key objectives clearly.",
    "Regular maintenance ensured smooth operation of equipment.",
    "The analysis revealed several interesting patterns.",
    "Participants shared their experiences and insights.",
    "The report summarized findings from the past quarter.",
    "Careful planning led to a successful implementation.",
]


class PasskeyRetrievalDataset(Dataset):
    """Dataset that generates passkey retrieval examples.

    Each example consists of filler text with a passkey inserted at a random position.
    The model must retrieve the passkey when prompted at the end.

    Format:
        [filler text...] The secret number is XXXXX. [more filler text...]
        What is the secret number? The secret number is
    """

    def __init__(
        self,
        num_samples: int = 10000,
        context_length: int = 2048,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.context_length = context_length
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained('gpt2')
        self.rng = random.Random(seed)

        # Pre-tokenize filler sentences
        self.filler_token_ids = [
            self.tokenizer.encode(" " + s) for s in FILLER_SENTENCES
        ]

        # Pre-tokenize the prompt template parts
        self.prompt_suffix_ids = self.tokenizer.encode(
            " What is the secret number? The secret number is"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Generate a single passkey retrieval example.

        Returns:
            dict with:
                - input_ids: (context_length,) tensor
                - labels: (context_length,) tensor (-100 for non-answer tokens)
                - passkey: str — the correct passkey
                - passkey_position: int — token position where passkey was inserted
        """
        rng = random.Random(self.rng.randint(0, 2**32) + idx)

        # Generate a random 5-digit passkey
        passkey = ''.join(rng.choices(string.digits, k=5))
        passkey_text = f" The secret number is {passkey}."
        passkey_ids = self.tokenizer.encode(passkey_text)
        answer_ids = self.tokenizer.encode(" " + passkey)

        # Calculate how many filler tokens we need
        # Reserve space for: passkey + prompt suffix + answer
        reserved = len(passkey_ids) + len(self.prompt_suffix_ids) + len(answer_ids)
        filler_budget = self.context_length - reserved

        # Generate filler tokens by repeating random sentences
        filler_ids = []
        while len(filler_ids) < filler_budget:
            sentence_ids = rng.choice(self.filler_token_ids)
            filler_ids.extend(sentence_ids)
        filler_ids = filler_ids[:filler_budget]

        # Choose a random insertion point for the passkey within the filler
        insert_pos = rng.randint(0, len(filler_ids))
        passkey_token_position = insert_pos

        # Build the full sequence
        # [filler_before] [passkey] [filler_after] [prompt_suffix] [answer]
        full_ids = (
            filler_ids[:insert_pos]
            + passkey_ids
            + filler_ids[insert_pos:]
            + self.prompt_suffix_ids
            + answer_ids
        )

        # Truncate or pad to exact context_length
        if len(full_ids) > self.context_length:
            full_ids = full_ids[:self.context_length]
        elif len(full_ids) < self.context_length:
            # Pad with EOS token
            pad_id = self.tokenizer.eos_token_id
            full_ids = full_ids + [pad_id] * (self.context_length - len(full_ids))

        input_ids = torch.tensor(full_ids, dtype=torch.long)

        # Labels: only supervise the answer tokens at the end
        # -100 = ignore index for cross-entropy loss
        labels = torch.full_like(input_ids, -100)
        # The answer starts after the prompt suffix
        answer_start = self.context_length - len(answer_ids)
        labels[answer_start:] = input_ids[answer_start:]

        return {
            'input_ids': input_ids,              # (context_length,)
            'labels': labels,                    # (context_length,)
            'passkey': passkey,
            'passkey_position': passkey_token_position,
        }


class PG19Dataset(Dataset):
    """PG19 long-document language modeling dataset.

    Loads books from the PG19 dataset and splits them into chunks of context_length.
    """

    def __init__(
        self,
        split: str = 'train',
        context_length: int = 2048,
        tokenizer: GPT2Tokenizer | None = None,
        max_books: int | None = None,
    ):
        from datasets import load_dataset

        self.context_length = context_length
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained('gpt2')

        # Load PG19
        dataset = load_dataset('pg19', split=split, trust_remote_code=True)
        if max_books is not None:
            dataset = dataset.select(range(min(max_books, len(dataset))))

        # Tokenize all books and concatenate, then split into chunks
        self.chunks = []
        for book in dataset:
            token_ids = self.tokenizer.encode(book['text'])
            # Split into non-overlapping chunks
            for i in range(0, len(token_ids) - context_length, context_length):
                self.chunks.append(token_ids[i:i + context_length])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        """Returns a chunk for language modeling.

        Returns:
            dict with:
                - input_ids: (context_length,) tensor
                - labels: (context_length,) tensor (shifted internally by the model)
        """
        input_ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }
