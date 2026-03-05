"""Character-level CNN following Zhang et al. (2015).

Architecture from CharCNN.ipynb in the reference cross-lingual benchmark:
  One-hot char encoding (68 chars, max 1500) → 6 Conv1d layers (256 filters)
  → 2 FC layers (1024) → output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 68-character vocabulary: lowercase letters, digits, punctuation
CHAR_VOCAB = list(
    "abcdefghijklmnopqrstuvwxyz0123456789"
    "-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}


def text_to_onehot(
    text: str,
    char_vocab: list = None,
    max_len: int = 1500,
) -> torch.Tensor:
    """Convert a raw string to a one-hot character tensor.

    Returns shape (len(char_vocab), max_len) — Conv1d expects (channels, length).
    """
    if char_vocab is None:
        char_vocab = CHAR_VOCAB
    c2i = {c: i for i, c in enumerate(char_vocab)}
    n = len(char_vocab)
    tensor = torch.zeros(n, max_len)
    for col, ch in enumerate(text.lower()[:max_len]):
        if ch in c2i:
            tensor[c2i[ch], col] = 1.0
    return tensor


class CharCNNClassifier(nn.Module):
    """6-layer character-level convolutional network.

    Uses SGD optimiser (lr=0.01) and 30 training epochs as in the paper.
    Max sequence length: 1500 characters.  Vocabulary: 68 characters.
    """

    def __init__(
        self,
        num_chars: int = 68,
        max_seq_len: int = 1500,
        num_classes: int = 14,
        n_conv_filters: int = 256,
        n_fc_neurons: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        # 6 convolutional layers; max-pool after layers 1, 2, 6
        def _conv_block(in_ch, out_ch, pool=False):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(kernel_size=3, stride=3))
            return nn.Sequential(*layers)

        self.conv1 = _conv_block(num_chars, n_conv_filters, pool=True)
        self.conv2 = _conv_block(n_conv_filters, n_conv_filters, pool=True)
        self.conv3 = _conv_block(n_conv_filters, n_conv_filters, pool=False)
        self.conv4 = _conv_block(n_conv_filters, n_conv_filters, pool=False)
        self.conv5 = _conv_block(n_conv_filters, n_conv_filters, pool=False)
        self.conv6 = _conv_block(n_conv_filters, n_conv_filters, pool=True)

        # Compute flattened size after conv stack
        conv_out_len = max_seq_len // (3 * 3 * 3)  # three max-pool layers, stride 3 each
        flat_size = n_conv_filters * conv_out_len

        self.fc = nn.Sequential(
            nn.Linear(flat_size, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_neurons, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_chars, max_seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class CharDataset(torch.utils.data.Dataset):
    """Dataset that converts text strings to one-hot char tensors."""

    def __init__(
        self,
        texts: list,
        labels: list,
        max_len: int = 1500,
        char_vocab: list = None,
    ):
        if char_vocab is None:
            char_vocab = CHAR_VOCAB
        self.tensors = [text_to_onehot(t, char_vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], torch.tensor(self.labels[idx], dtype=torch.long)
