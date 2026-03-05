"""Bidirectional GRU text classifier with pre-trained Word2Vec embeddings.

Architecture from BiGRU.ipynb in the reference cross-lingual benchmark:
  Embedding(vocab, 50) → BiGRU(50, 256, 2 layers) → Dropout → Linear(512, num_classes)
"""

import torch
import torch.nn as nn


class BiGRUClassifier(nn.Module):
    """Bidirectional multi-layer GRU for text classification.

    Best traditional model in the Paper 1 benchmark: 83.32% accuracy on
    Kirundi after fine-tuning from Kinyarwanda.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        hidden_dim: int = 256,
        output_dim: int = 14,
        n_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: (batch, seq_len)
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.rnn(embedded)
        # hidden: (num_layers * 2, batch, hidden_dim) — take last layer
        hidden = self.dropout(
            torch.cat((hidden[-2], hidden[-1]), dim=1)
        )
        return self.fc(hidden)

    def load_pretrained_embeddings(self, vectors: torch.Tensor) -> None:
        """Initialise embedding layer from a pre-trained weight matrix."""
        assert vectors.shape == self.embedding.weight.data.shape, (
            f"shape mismatch: got {vectors.shape}, "
            f"expected {self.embedding.weight.data.shape}"
        )
        self.embedding.weight.data.copy_(vectors)


def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(iterator), correct / total


def evaluate_epoch(model, iterator, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(iterator), correct / total
