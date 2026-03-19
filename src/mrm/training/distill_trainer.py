"""Custom training loop for knowledge distillation (Paper 3).

Implements the two-stage training procedure from the AfroXLMR-Comet paper:
  Stage 1: Distillation — teacher frozen, student trained with KL + attention MSE
  Stage 2: Fine-tuning — student fine-tuned on each language's AfriSenti data
"""

import logging
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mrm.models.distillation import AttentionProjection, distillation_loss
from mrm.training.callbacks import EarlyStopping
from mrm.evaluation.metrics import weighted_f1

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Teacher → student knowledge distillation with FP16 and gradient accumulation.

    Hyperparameters from Paper 3:
        lr = 5e-5, epochs = 15, batch_size = 8, grad_accum = 2
        temperature = 2.0, alpha = 0.5, patience = 3
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        seq_len: int = 128,
        temperature: float = 2.0,
        alpha: float = 0.5,
        lr: float = 5e-5,
        num_epochs: int = 15,
        grad_accum_steps: int = 2,
        patience: int = 3,
        fp16: bool = True,
        device: str = "cuda",
        output_dir: str = "outputs/distillation",
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader
        self.temperature = temperature
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.grad_accum = grad_accum_steps
        self.fp16 = fp16 and device.startswith("cuda")
        self.device = device
        self.output_dir = output_dir

        self.attn_proj = AttentionProjection(seq_len=seq_len).to(device)
        self.optimizer = torch.optim.AdamW(
            list(student_model.parameters()) + list(self.attn_proj.parameters()),
            lr=lr,
        )
        self.scaler = GradScaler() if self.fp16 else None
        self.early_stopping = EarlyStopping(patience=patience, metric="f1")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _distillation_step(self, batch: dict) -> torch.Tensor:
        labels = batch.get("labels")
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}

        with torch.no_grad():
            teacher_out = self.teacher(**inputs, output_attentions=True)

        if self.fp16:
            with autocast():
                student_out = self.student(**inputs, output_attentions=True)
                loss, l_d, l_a = distillation_loss(
                    student_out.logits,
                    teacher_out.logits,
                    student_out.attentions,
                    teacher_out.attentions,
                    self.attn_proj,
                    self.temperature,
                    self.alpha,
                )
        else:
            student_out = self.student(**inputs, output_attentions=True)
            loss, l_d, l_a = distillation_loss(
                student_out.logits,
                teacher_out.logits,
                student_out.attentions,
                teacher_out.attentions,
                self.attn_proj,
                self.temperature,
                self.alpha,
            )

        return loss, l_d, l_a

    def train(self) -> dict:
        """Run full distillation training. Returns per-epoch history."""
        history = {"loss": [], "eval_f1": []}

        for epoch in range(self.num_epochs):
            self.student.train()
            self.attn_proj.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_dl):
                loss, l_d, l_a = self._distillation_step(batch)
                loss = loss / self.grad_accum

                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.grad_accum == 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.grad_accum

            eval_metrics = self.evaluate()
            history["loss"].append(epoch_loss / len(self.train_dl))
            history["eval_f1"].append(eval_metrics["f1"])

            logger.info(
                "epoch %d/%d — loss %.4f | eval_f1 %.4f",
                epoch + 1, self.num_epochs,
                history["loss"][-1], eval_metrics["f1"],
            )

            if self.early_stopping(eval_metrics):
                logger.info("early stopping triggered at epoch %d", epoch + 1)
                break

        return history

    def evaluate(self) -> dict:
        self.student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.eval_dl:
                labels = batch.pop("labels")
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.student(**inputs).logits
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.tolist())
        return {
            "f1": weighted_f1(all_labels, all_preds),
        }

    def save_student(self, path: str = None):
        out = path or f"{self.output_dir}/student"
        self.student.model.save_pretrained(out)
        logger.info("saved student model → %s", out)
