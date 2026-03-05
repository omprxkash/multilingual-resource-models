"""PyTorch Dataset wrappers for transformer and embedding-based pipelines."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


class HFTextDataset(Dataset):
    """Tokeniser-encoded dataset for HuggingFace transformer models.

    Works with any AutoTokenizer and handles padding/truncation internally.
    Used by Paper 1 transformer experiments, Paper 2 MAGE, and Paper 3 distillation.
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    @classmethod
    def from_csv(cls, path: str, tokenizer, max_length: int = 128, text_col: str = "text", label_col: str = "label"):
        df = pd.read_csv(path)
        texts = df[text_col].fillna("").tolist()
        labels = df[label_col].tolist()
        return cls(texts, labels, tokenizer, max_length)


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed sentence embeddings + labels.

    Used in Paper 2 MAGE pipeline: embeddings are extracted from AfriBERTa
    once, stored as tensors, then fed to MAGE/LSTM/LR classifiers.
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        assert len(embeddings) == len(labels), "embeddings and labels must have the same length"
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self.embeddings[idx], self.labels[idx]

    @classmethod
    def from_numpy(cls, embeddings: np.ndarray, labels: np.ndarray):
        return cls(
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


class SimpleVocabDataset(Dataset):
    """Word-index encoded dataset for BiGRU / TextCNN.

    Builds vocabulary from training data, encodes sequences to fixed length.
    Replaces the legacy torchtext 0.6 dependency used in the original notebooks.
    """

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(
        self,
        texts: list,
        labels: list,
        vocab: dict = None,
        max_length: int = 256,
    ):
        self.labels = labels
        self.max_length = max_length

        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

        self.encoded = [self._encode(t) for t in texts]

    def _build_vocab(self, texts: list) -> dict:
        from collections import Counter
        import nltk
        counter = Counter()
        for t in texts:
            counter.update(str(t).lower().split())
        vocab = {self.PAD: 0, self.UNK: 1}
        for word, _ in counter.most_common(15000):
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _encode(self, text: str) -> list:
        tokens = str(text).lower().split()[: self.max_length]
        ids = [self.vocab.get(t, self.vocab[self.UNK]) for t in tokens]
        # pad or truncate
        ids = ids + [self.vocab[self.PAD]] * (self.max_length - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @classmethod
    def build_pair(cls, train_path: str, test_path: str, max_length: int = 256, text_col: str = "text", label_col: str = "label"):
        """Build train and test datasets sharing the same vocabulary."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_texts = train_df[text_col].fillna("").tolist()
        train_labels = train_df[label_col].tolist()
        train_ds = cls(train_texts, train_labels, vocab=None, max_length=max_length)
        test_texts = test_df[text_col].fillna("").tolist()
        test_labels = test_df[label_col].tolist()
        test_ds = cls(test_texts, test_labels, vocab=train_ds.vocab, max_length=max_length)
        return train_ds, test_ds

    def load_pretrained_embeddings(self, embedding_path: str, vector_size: int = 50) -> torch.Tensor:
        """Load GloVe-format embedding file and return weight matrix aligned to vocab."""
        import numpy as np
        emb = np.zeros((len(self.vocab), vector_size), dtype=np.float32)
        found = 0
        with open(embedding_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split()
                word = parts[0]
                if word in self.vocab:
                    emb[self.vocab[word]] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        return torch.tensor(emb, dtype=torch.float32)
