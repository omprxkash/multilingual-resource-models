"""HuggingFace transformer classifiers for cross-lingual transfer (Paper 1).

Wraps AutoModelForSequenceClassification for mBERT, AfriBERTa, BantuBERTa,
and AfroXLMR-Large with a shared training-args factory.
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

SUPPORTED_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "afriberta": "castorini/afriberta_large",
    "bantubert": "dsfsi/BantuBERTa",
    "afroxlmr": "Davlan/afro-xlmr-large",
}


def load_transformer_classifier(
    model_name: str,
    num_labels: int,
    checkpoint_path: str = None,
) -> tuple:
    """Load model and tokenizer.

    Args:
        model_name: key in SUPPORTED_MODELS or a direct HuggingFace model ID.
        num_labels:  number of classification labels.
        checkpoint_path: local directory to load fine-tuned weights from.

    Returns:
        (model, tokenizer) tuple ready for Trainer or manual training.
    """
    hf_name = SUPPORTED_MODELS.get(model_name, model_name)
    source = checkpoint_path if checkpoint_path else hf_name

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def make_training_args(
    output_dir: str,
    num_epochs: int = 8,
    batch_size: int = 32,
    lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    fp16: bool = False,
    logging_steps: int = 50,
    save_total_limit: int = 2,
) -> TrainingArguments:
    """Build HuggingFace TrainingArguments with the Paper 1 defaults."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=lr,
        fp16=fp16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        report_to="none",
    )
