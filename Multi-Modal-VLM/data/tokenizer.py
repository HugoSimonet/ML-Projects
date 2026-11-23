"""
Text tokenizer utilities
"""

import re
from typing import List, Dict, Optional
import json
import os


class SimpleTokenizer:
    """
    Simple word-level tokenizer for text processing
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_length: int = 77,
        pad_token: str = '[PAD]',
        unk_token: str = '[UNK]',
        bos_token: str = '[CLS]',
        eos_token: str = '[SEP]',
        lowercase: bool = True
    ):
        """
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            lowercase: Convert text to lowercase
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lowercase = lowercase

        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Initialize vocabulary
        self.vocab = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 101,
            eos_token: 102
        }

        self.id2token = {v: k for k, v in self.vocab.items()}

        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]
        self.bos_token_id = self.vocab[bos_token]
        self.eos_token_id = self.vocab[eos_token]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.lowercase:
            text = text.lower()

        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length (uses self.max_length if None)
            padding: Pad to max_length
            truncation: Truncate to max_length

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        max_length = max_length or self.max_length

        # Tokenize
        tokens = self.tokenize(text)

        # Convert to IDs
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Truncate
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length - 1] + [self.eos_token_id]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad
        if padding:
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []
        special_tokens = {self.pad_token_id, self.bos_token_id, self.eos_token_id}

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_tokens:
                continue
            token = self.id2token.get(token_id, self.unk_token)
            tokens.append(token)

        return ' '.join(tokens)

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from texts

        Args:
            texts: List of texts to build vocabulary from
        """
        word_freq = {}

        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Add to vocabulary
        current_id = max(self.vocab.values()) + 1
        for word, _ in sorted_words:
            if word not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[word] = current_id
                self.id2token[current_id] = word
                current_id += 1

    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'lowercase': self.lowercase
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data['vocab_size'],
            max_length=data['max_length'],
            pad_token=data['pad_token'],
            unk_token=data['unk_token'],
            bos_token=data['bos_token'],
            eos_token=data['eos_token'],
            lowercase=data['lowercase']
        )

        tokenizer.vocab = data['vocab']
        tokenizer.id2token = {v: k for k, v in tokenizer.vocab.items()}

        return tokenizer

    def __len__(self):
        return len(self.vocab)


def get_tokenizer(tokenizer_type: str = 'simple', **kwargs):
    """
    Get tokenizer instance

    Args:
        tokenizer_type: Type of tokenizer ('simple' or 'bert')
        **kwargs: Additional arguments for tokenizer

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == 'simple':
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type == 'bert':
        try:
            from transformers import BertTokenizer
            model_name = kwargs.get('model_name', 'bert-base-uncased')
            return BertTokenizer.from_pretrained(model_name)
        except ImportError:
            print("transformers not installed, falling back to simple tokenizer")
            return SimpleTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SimpleTokenizer()

    # Build vocab from sample texts
    texts = [
        "A cat sitting on a chair",
        "A dog playing in the park",
        "The sun is shining brightly"
    ]
    tokenizer.build_vocab(texts)

    # Test encoding
    text = "A cat sitting on a chair"
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")

    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")

    # Test save/load
    tokenizer.save('test_tokenizer.json')
    loaded_tokenizer = SimpleTokenizer.load('test_tokenizer.json')
    print(f"Loaded tokenizer vocab size: {len(loaded_tokenizer)}")
