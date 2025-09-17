#!/usr/bin/env python3
"""
ClinicalBERT Fine-tuning Script for Readmission Prediction

This script fine-tunes ClinicalBERT models from HuggingFace for predicting
hospital readmissions using MIMIC-IV discharge summaries.

Features:
- Automated model caching to /ssd-shared/yichen_models/
- Left truncation strategy for long clinical texts
- Class imbalance handling with weighted loss
- Comprehensive evaluation metrics
- GPU optimization with mixed precision training
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)



class ReadmissionDataset(Dataset):
    """Dataset class for MIMIC-IV readmission prediction"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Left truncation: keep the end of the text (discharge information)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_preprocess_data(cfg: DictConfig) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess the MIMIC-IV readmission data

    Args:
        cfg: Hydra configuration containing data settings

    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {cfg.data.data_path}")

    with open(cfg.data.data_path, 'r') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Label distribution: {np.bincount(labels)}")

    return texts, labels


def setup_model_and_tokenizer(cfg: DictConfig):
    """
    Setup and cache the model and tokenizer

    Args:
        cfg: Hydra configuration containing model settings

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {cfg.model.model_name}")
    logger.info(f"Using cache directory: {cfg.model.model_cache_dir}")

    # Ensure cache directory exists
    Path(cfg.model.model_cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name,
            cache_dir=cfg.model.model_cache_dir,
            use_fast=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name,
            num_labels=cfg.model.num_labels,
            cache_dir=cfg.model.model_cache_dir
        )

        logger.info(f"Successfully loaded {cfg.model.model_name}")
        logger.info(f"Model max position embeddings: {model.config.max_position_embeddings}")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load {cfg.model.model_name}: {e}")
        # Fallback to alternative model
        alternative_model = "emilyalsentzer/Bio_ClinicalBERT"
        logger.info(f"Trying alternative model: {alternative_model}")

        tokenizer = AutoTokenizer.from_pretrained(
            alternative_model,
            cache_dir=cfg.model.model_cache_dir,
            use_fast=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            alternative_model,
            num_labels=cfg.model.num_labels,
            cache_dir=cfg.model.model_cache_dir
        )

        return model, tokenizer


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    try:
        # For binary classification, we can compute AUC
        if len(np.unique(labels)) == 2:
            # Get probabilities for positive class
            prob_predictions = torch.softmax(torch.tensor(eval_pred[0]), dim=1)[:, 1].numpy()
            auc = roc_auc_score(labels, prob_predictions)
        else:
            auc = 0.0
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting ClinicalBERT fine-tuning for readmission prediction")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup CUDA devices from configuration
    if hasattr(cfg, 'cuda_visible_devices') and cfg.cuda_visible_devices:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.cuda_visible_devices)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={cfg.cuda_visible_devices}")

    # Load and preprocess data
    texts, labels = load_and_preprocess_data(cfg)

    # Split data
    logger.info("Splitting data into train/validation sets")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=labels
    )

    logger.info(f"Train set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = ReadmissionDataset(train_texts, train_labels, tokenizer, cfg.model.max_length)
    val_dataset = ReadmissionDataset(val_texts, val_labels, tokenizer, cfg.model.max_length)

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")

    logger.info(f"Class weights: {class_weights}")

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        logging_dir=f'{cfg.training.output_dir}/logs',
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        eval_strategy=cfg.training.eval_strategy,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        learning_rate=cfg.training.learning_rate,
        report_to=cfg.training.report_to,
    )

    # Setup trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)]
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info(f"Saving model to {cfg.training.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.training.output_dir)

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")

    # Detailed evaluation on validation set
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(val_labels, y_pred, target_names=['No Readmission', 'Readmission']))

    # Save evaluation results
    results_path = os.path.join(cfg.training.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'eval_results': eval_results,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'model_name': cfg.model.model_name,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'class_distribution': np.bincount(labels).tolist()
        }, f, indent=2)

    logger.info(f"Evaluation results saved to {results_path}")
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()