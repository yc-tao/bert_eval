#!/usr/bin/env python3
"""
ClinicalBERT Fine-tuning Script for Readmission Prediction (Modularized)

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
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.loader import load_and_preprocess_data, split_data
from src.data.dataset import ReadmissionDataset
from src.models.model_manager import ModelManager
from src.training.trainer import WeightedTrainer, compute_class_weights, setup_training_args
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting ClinicalBERT fine-tuning for readmission prediction")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Load and preprocess data
    texts, labels = load_and_preprocess_data(cfg.data.data_path)

    # Split data
    logger.info("Splitting data into train/validation sets")
    train_texts, val_texts, train_labels, val_labels = split_data(
        texts, labels,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )

    logger.info(f"Train set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")

    # Setup model and tokenizer using ModelManager
    model_manager = ModelManager(cfg.model.model_cache_dir)

    try:
        model, tokenizer = model_manager.load_model_and_tokenizer(
            cfg.model.model_name,
            cfg.model.num_labels
        )
    except Exception as e:
        logger.warning(f"Failed to load primary model {cfg.model.model_name}: {e}")
        logger.info("Attempting to load with fallback models")
        model, tokenizer = model_manager.load_with_fallback(
            cfg.model.model_name,
            cfg.model.num_labels
        )

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = ReadmissionDataset(train_texts, train_labels, tokenizer, cfg.model.max_length)
    val_dataset = ReadmissionDataset(val_texts, val_labels, tokenizer, cfg.model.max_length)

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_labels)
    logger.info(f"Class weights: {class_weights}")

    # Setup training arguments
    training_args = setup_training_args(cfg.training, cfg.training.output_dir)

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