#!/usr/bin/env python3
"""
Custom training utilities for ClinicalBERT
"""

import torch
import numpy as np
import logging
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced data and multi-GPU support"""

    def __init__(self, class_weights=None, device_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.device_manager = device_manager

        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            self._setup_multi_gpu()

    def _setup_multi_gpu(self):
        """Setup multi-GPU configuration"""
        device_count = torch.cuda.device_count()

        # Use DataParallel for simple multi-GPU setup
        if not hasattr(self.model, 'module'):
            logger.info(f"Wrapping model with DataParallel for {device_count} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # Ensure class weights are on the correct device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.model.device if hasattr(self.model, 'device') else 'cuda')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.class_weights is not None:
            # Ensure class weights are on the same device as logits
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)

            # Apply class weights
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)

            # Handle DataParallel case where we need to get the actual model config
            if hasattr(model, 'module'):
                num_labels = model.module.config.num_labels
            else:
                num_labels = model.config.num_labels

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """Override training step to monitor GPU memory if device manager is available"""
        result = super().training_step(model, inputs)

        # Log GPU memory usage periodically
        if (self.device_manager and
            hasattr(self.state, 'global_step') and
            self.state.global_step % 100 == 0):
            memory_info = self.device_manager.monitor_gpu_memory()
            for device_id, info in memory_info.items():
                logger.debug(f"GPU {device_id}: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
                            f"({info['utilization']*100:.1f}%)")

        return result


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


def setup_training_args(cfg, output_dir: str, device_manager=None):
    """
    Setup training arguments from configuration

    Args:
        cfg: Training configuration
        output_dir: Output directory for training
        device_manager: DeviceManager instance for multi-GPU setup

    Returns:
        TrainingArguments object
    """
    from transformers import TrainingArguments

    # Get device configuration
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = device_count > 1

    # Adjust batch sizes for multi-GPU if device manager is provided
    train_batch_size = cfg.train_batch_size
    eval_batch_size = cfg.eval_batch_size
    gradient_accumulation_steps = getattr(cfg, 'gradient_accumulation_steps', 1)

    if device_manager and use_multi_gpu:
        # Optionally adjust batch size per device
        train_batch_size = device_manager.get_optimal_batch_size(cfg.train_batch_size, device_count)
        eval_batch_size = device_manager.get_optimal_batch_size(cfg.eval_batch_size, device_count)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        dataloader_pin_memory=True,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        learning_rate=cfg.learning_rate,
        report_to=cfg.report_to,
        # Multi-GPU settings
        ddp_find_unused_parameters=False if use_multi_gpu else None,
        remove_unused_columns=False,
    )

    if use_multi_gpu:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Multi-GPU training enabled with {device_count} devices")
        logger.info(f"Effective batch size: {train_batch_size * device_count * gradient_accumulation_steps}")

    return training_args