"""MAGE: Multi-Head Attention Guided Embeddings — Paper 2.

The MAGE classifier takes pooled sentence embeddings (768-dim from AfriBERTa)
and applies 4-head attention with learnable context vectors before classification.
Also includes the LSTM baseline and a logistic regression wrapper used for ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression as _SKLearnLR
import numpy as np


class MAGEClassifier(nn.Module):
    """Multi-Head Attention Gate Ensemble classifier.

    Each of the 4 attention heads has a learnable context vector that
    scores the input embedding.  The softmax-weighted embedding outputs
    from all heads are concatenated then summed, producing a refined
    representation for classification.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Learnable context vectors — one per head
        self.context_vectors = nn.Parameter(torch.randn(num_heads, embed_dim))
        nn.init.xavier_uniform_(self.context_vectors.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (batch, embed_dim) → logits (batch, num_classes)."""
        # Each head independently scores and weights the embedding
        head_outputs = []
        for h in range(self.num_heads):
            ctx = self.context_vectors[h]                       # (embed_dim,)
            score = (embeddings * ctx).sum(dim=-1, keepdim=True)  # (batch, 1)
            weight = torch.sigmoid(score)                        # soft gate
            head_out = embeddings * weight                       # (batch, embed_dim)
            head_outputs.append(head_out)
        # Concatenate across heads then sum to get final representation
        stacked = torch.stack(head_outputs, dim=1)              # (batch, heads, embed_dim)
        aggregated = stacked.sum(dim=1)                         # (batch, embed_dim)
        aggregated = self.dropout(aggregated)
        return self.classifier(aggregated)


class LSTMClassifier(nn.Module):
    """LSTM baseline for Paper 2 sentiment classification.

    Takes a sequence of embeddings (batch, seq_len, 768) and uses the
    final hidden state for classification.  lr=0.001, hidden_dim=128.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, embed_dim) — embeddings are treated as single-step sequences."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n.squeeze(0)))


class LogisticRegressionWrapper:
    """Scikit-learn LogisticRegression wrapped for the MAGE pipeline.

    Fits on numpy arrays extracted from the embedding dataset.
    Uses LBFGS solver with max_iter=1000 per the paper.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self.model = _SKLearnLR(solver="lbfgs", max_iter=max_iter, C=C, multi_class="auto")

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        self.model.fit(X, y)
        return self

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.model.predict(X)

    def score(self, embeddings, labels) -> float:
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        return float(self.model.score(X, y))
