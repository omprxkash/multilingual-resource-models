"""TextCNN classifier following Kim (2014) with multiple filter sizes.

Architecture from CNN.ipynb in the reference cross-lingual benchmark:
  Embedding(vocab, 50) → Conv2d(1, 150, [3,4,5]) → MaxPool → Dropout → Linear(450, 14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNClassifier(nn.Module):
    """Convolutional neural network for sentence classification.

    Uses parallel convolutional filters of sizes 3, 4, 5 to capture
    n-gram features at different granularities.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        n_filters: int = 150,
        filter_sizes: list = None,
        output_dim: int = 14,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(fs, embedding_dim),
            )
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: (batch, seq_len)
        embedded = self.embedding(text).unsqueeze(1)  # (batch, 1, seq_len, emb_dim)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

    def load_pretrained_embeddings(self, vectors: torch.Tensor) -> None:
        self.embedding.weight.data.copy_(vectors)


def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(iterator), correct / total


def evaluate_epoch(model, iterator, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(iterator), correct / total
