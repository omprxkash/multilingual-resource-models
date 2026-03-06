"""Language-Independent Data Augmentation (LiDA) — Paper 2 (MAGE).

LiDA operates entirely in the embedding space: it adds uniform random noise
to sentence embeddings, producing augmented training examples without any
language-specific resources (no WordNet, no back-translation dictionaries).
This makes it effective for Bantu languages with minimal NLP tooling.
"""

import torch
from torch.utils.data import DataLoader


class LiDA:
    """Language-Independent Data Augmentation via embedding-space noise.

    Adds noise sampled uniformly from [r_min, r_max] to each dimension of
    each embedding vector.  The range controls augmentation strength:
    small values preserve semantics; larger values increase diversity.
    """

    def __init__(self, r_min: float = 0.0, r_max: float = 0.1):
        assert r_min <= r_max, "r_min must be <= r_max"
        self.r_min = r_min
        self.r_max = r_max

    def augment(
        self,
        embeddings: torch.Tensor,
        n_augmented: int = 1,
    ) -> torch.Tensor:
        """Apply noise to a batch of embeddings.

        Args:
            embeddings: (batch_size, embed_dim) float tensor.
            n_augmented: number of augmented copies per original.

        Returns:
            (batch_size * n_augmented, embed_dim) augmented tensor.
        """
        copies = []
        for _ in range(n_augmented):
            noise = torch.zeros_like(embeddings).uniform_(self.r_min, self.r_max)
            copies.append(embeddings + noise)
        return torch.cat(copies, dim=0)

    def augment_dataset(
        self,
        embedding_model,
        dataloader: DataLoader,
        device: str = "cpu",
        n_augmented: int = 1,
    ) -> tuple:
        """Extract embeddings via model, apply LiDA, return augmented tensors.

        The embedding model is expected to accept tokenizer output dicts and
        return a HuggingFace ModelOutput with a `last_hidden_state` field.
        Mean-pooling over the sequence dimension gives the sentence embedding.

        Returns:
            (augmented_embeddings, repeated_labels) as float/long tensors.
        """
        embedding_model.eval().to(device)
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                output = embedding_model(**inputs, output_hidden_states=True)
                # mean-pool the last hidden state over the sequence dimension
                emb = output.last_hidden_state.mean(dim=1)
                all_embeddings.append(emb.cpu())
                all_labels.append(labels.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        aug_embeddings = self.augment(embeddings, n_augmented)
        aug_labels = labels.repeat(n_augmented)

        combined_embeddings = torch.cat([embeddings, aug_embeddings], dim=0)
        combined_labels = torch.cat([labels, aug_labels], dim=0)

        return combined_embeddings, combined_labels
