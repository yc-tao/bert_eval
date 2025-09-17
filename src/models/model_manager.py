#!/usr/bin/env python3
"""
Model management utilities for ClinicalBERT
"""

import logging
from typing import Tuple, Optional
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, caching, and setup"""

    def __init__(self, cache_dir: str = "/ssd-shared/yichen_models"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def load_model_and_tokenizer(self, model_name: str, num_labels: int = 2,
                                use_fast_tokenizer: bool = True, device: str = None) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load model and tokenizer from HuggingFace or local path

        Args:
            model_name: Model name or path
            num_labels: Number of classification labels
            use_fast_tokenizer: Whether to use fast tokenizer
            device: Device to load model on (auto-detected if None)

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model and tokenizer: {model_name}")
        logger.info(f"Using cache directory: {self.cache_dir}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                use_fast=use_fast_tokenizer
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=self.cache_dir
            )

            logger.info(f"Successfully loaded {model_name}")
            if hasattr(model.config, 'max_position_embeddings'):
                logger.info(f"Model max position embeddings: {model.config.max_position_embeddings}")

            # Move model to specified device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            if torch.cuda.is_available() and device.startswith("cuda"):
                model = model.to(device)
                logger.info(f"Model moved to device: {device}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise

    def load_pretrained_model(self, model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load a locally fine-tuned model

        Args:
            model_path: Path to the fine-tuned model directory

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading pretrained model from: {model_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            logger.info(f"Successfully loaded pretrained model from {model_path}")
            logger.info(f"Model config: {model.config}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load pretrained model from {model_path}: {e}")
            raise

    def get_model_info(self, model_name_or_path: str) -> dict:
        """
        Get information about a model without loading it

        Args:
            model_name_or_path: Model name or path

        Returns:
            Dictionary with model information
        """
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=self.cache_dir)

            info = {
                'model_type': config.model_type,
                'num_labels': getattr(config, 'num_labels', None),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get model info for {model_name_or_path}: {e}")
            return {}

    def setup_model_for_evaluation(self, model_name_or_path: str,
                                 device: Optional[str] = None) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Setup model and tokenizer for evaluation

        Args:
            model_name_or_path: Model name or local path
            device: Device to load model on (auto-detected if None)

        Returns:
            Tuple of (model, tokenizer) ready for evaluation
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if it's a local path or HuggingFace model
        if Path(model_name_or_path).exists():
            model, tokenizer = self.load_pretrained_model(model_name_or_path)
        else:
            # Try to determine number of labels from config
            info = self.get_model_info(model_name_or_path)
            num_labels = info.get('num_labels', 2)
            model, tokenizer = self.load_model_and_tokenizer(model_name_or_path, num_labels)

        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()

        logger.info(f"Model loaded on {device} and set to evaluation mode")

        return model, tokenizer

    def get_fallback_models(self) -> list:
        """
        Get list of fallback ClinicalBERT models

        Returns:
            List of model names to try as fallbacks
        """
        return [
            "medicalai/ClinicalBERT",
            "emilyalsentzer/Bio_ClinicalBERT",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ]

    def load_with_fallback(self, primary_model: str, num_labels: int = 2, device: str = None) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load model with fallback options

        Args:
            primary_model: Primary model to try
            num_labels: Number of classification labels
            device: Device to load model on

        Returns:
            Tuple of (model, tokenizer)
        """
        models_to_try = [primary_model] + self.get_fallback_models()

        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load {model_name}")
                return self.load_model_and_tokenizer(model_name, num_labels, device=device)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        raise RuntimeError("Failed to load any model, including fallbacks")