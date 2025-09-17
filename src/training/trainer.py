#!/usr/bin/env python3
"""
Custom training utilities for ClinicalBERT
"""

import torch
import numpy as np
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight


class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced data"""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.class_weights is not None:
            # Apply class weights
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels, device=None):
    """
    Compute class weights for imbalanced dataset

    Args:
        labels: List or array of labels
        device: Device to put weights on (auto-detected if None)

    Returns:
        Tensor of class weights
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    if device == "cuda" and torch.cuda.is_available():
        class_weights = class_weights.cuda()

    return class_weights


def setup_training_args(cfg, output_dir: str):
    """
    Setup training arguments from configuration

    Args:
        cfg: Training configuration
        output_dir: Output directory for training

    Returns:
        TrainingArguments object
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_dir=f'{output_dir}/logs',
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        dataloader_num_workers=cfg.dataloader_num_workers,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        learning_rate=cfg.learning_rate,
        report_to=cfg.report_to,
    )