"""AfroXLMR-Comet: compact student model for knowledge distillation — Paper 3.

Teacher: Davlan/afro-xlmr-large (559.9M params, 2.09 GB, 24 layers)
Student: AfroXLMR-Comet (68.9M params, 263 MB, 6 layers)

Hybrid distillation loss:
  L_distill = KLDiv(softmax(student/T), softmax(teacher/T))   — response-based
  L_attn    = MSE(project(student_attn), teacher_attn)        — feature-based
  L_total   = 0.5 * L_distill + 0.5 * L_attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

TEACHER_MODEL = "Davlan/afro-xlmr-large"

STUDENT_CONFIG = {
    "num_hidden_layers": 6,
    "hidden_size": 256,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
}


class AfroXLMRComet(nn.Module):
    """Compact 6-layer XLM-R student model.

    Initialized with a modified configuration derived from the teacher's
    tokenizer and embedding vocabulary, then trained via soft-label KL
    divergence and attention-map MSE distillation.

    Specs vs. teacher:
        hidden_size:       256 vs 1024  (75% smaller)
        attention_heads:   8   vs 16
        layers:            6   vs 24
        intermediate_size: 1024 vs 4096
        total parameters:  ~68.9M vs 559.9M  (87.7% reduction)
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
        )

    @classmethod
    def from_scratch(
        cls,
        teacher_name: str = TEACHER_MODEL,
        num_labels: int = 3,
    ) -> "AfroXLMRComet":
        """Build student config from teacher and initialize a fresh model."""
        teacher_cfg = AutoConfig.from_pretrained(teacher_name)
        student_cfg = AutoConfig.from_pretrained(teacher_name)

        # Override architecture with compact student dimensions
        for key, val in STUDENT_CONFIG.items():
            setattr(student_cfg, key, val)
        student_cfg.num_labels = num_labels

        hf_model = AutoModelForSequenceClassification.from_config(student_cfg)
        return cls(hf_model)

    def get_tokenizer(self, teacher_name: str = TEACHER_MODEL):
        return AutoTokenizer.from_pretrained(teacher_name)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AttentionProjection(nn.Module):
    """Linear projection to align student and teacher attention dimensions.

    Flattened student attention (seq_len × seq_len in student space) is
    projected to match the teacher's flattened attention for MSE comparison.
    """

    def __init__(
        self,
        seq_len: int = 128,
        student_heads: int = 8,
        teacher_heads: int = 16,
    ):
        super().__init__()
        student_flat = seq_len * seq_len
        teacher_flat = seq_len * seq_len
        # Project from student mean-pooled attention to teacher size
        self.proj = nn.Linear(student_flat, teacher_flat)

    def forward(self, student_attn: torch.Tensor) -> torch.Tensor:
        """student_attn: (batch, heads, seq, seq) → projected flat tensor."""
        # Mean across attention heads → (batch, seq, seq)
        mean_attn = student_attn.mean(dim=1)
        # Flatten → (batch, seq * seq)
        flat = mean_attn.flatten(start_dim=1)
        return self.proj(flat)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_attentions: list,
    teacher_attentions: list,
    attention_proj: nn.Module,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple:
    """Compute combined knowledge distillation loss.

    Args:
        student_logits:   (batch, num_labels) raw student logits.
        teacher_logits:   (batch, num_labels) raw teacher logits (no grad).
        student_attentions: list of (batch, heads, seq, seq) attention tensors.
        teacher_attentions: list of (batch, heads, seq, seq) attention tensors.
        attention_proj:   AttentionProjection module for student→teacher dim.
        temperature:      softmax temperature T — higher = softer targets.
        alpha:            weight for distillation loss (1-alpha for attention).

    Returns:
        (L_total, L_distill, L_attn) tuple.
    """
    # Response-based: KL divergence between soft probability distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    L_distill = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Feature-based: MSE between last-layer mean-pooled attention maps
    s_attn = student_attentions[-1]   # last student layer
    t_attn = teacher_attentions[-1]   # last teacher layer
    s_proj = attention_proj(s_attn)
    # Teacher: mean-pool heads, flatten
    t_mean = t_attn.mean(dim=1).flatten(start_dim=1)
    L_attn = F.mse_loss(s_proj, t_mean.detach())

    L_total = alpha * L_distill + (1.0 - alpha) * L_attn
    return L_total, L_distill, L_attn
