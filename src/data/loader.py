#!/usr/bin/env python3
"""
Data loading utilities for ClinicalBERT fine-tuning and evaluation
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess the MIMIC-IV readmission data

    Args:
        data_path: Path to the JSON data file

    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Label distribution: {np.bincount(labels)}")

    return texts, labels


def split_data(texts: List[str], labels: List[int],
               test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Split data into train and validation sets

    Args:
        texts: List of text samples
        labels: List of corresponding labels
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_texts, val_texts, train_labels, val_labels)
    """
    logger.info("Splitting data into train/validation sets")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    logger.info(f"Train set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")

    return train_texts, val_texts, train_labels, val_labels


def load_evaluation_data(data_path: str, has_labels: bool = True) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load data for evaluation (may or may not have labels)

    Args:
        data_path: Path to the data file
        has_labels: Whether the data includes labels

    Returns:
        Tuple of (texts, labels) where labels is None if has_labels=False
    """
    logger.info(f"Loading evaluation data from {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    texts = []
    labels = []

    for item in data:
        if isinstance(item, dict):
            texts.append(item['text'])
            if has_labels and 'label' in item:
                labels.append(item['label'])
        else:
            # Handle case where data is just a list of strings
            texts.append(str(item))

    logger.info(f"Loaded {len(texts)} samples for evaluation")

    if has_labels and labels:
        logger.info(f"Label distribution: {np.bincount(labels)}")
        return texts, labels
    else:
        logger.info("No labels found or labels not expected")
        return texts, None


def get_data_statistics(texts: List[str], labels: Optional[List[int]] = None) -> Dict:
    """
    Get basic statistics about the dataset

    Args:
        texts: List of text samples
        labels: Optional list of labels

    Returns:
        Dictionary with dataset statistics
    """
    text_lengths = [len(text) for text in texts]

    stats = {
        'num_samples': len(texts),
        'text_length_stats': {
            'min': min(text_lengths),
            'max': max(text_lengths),
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'std': np.std(text_lengths)
        }
    }

    if labels is not None:
        label_counts = np.bincount(labels)
        stats['label_distribution'] = label_counts.tolist()
        stats['class_balance'] = {
            f'class_{i}': count / len(labels)
            for i, count in enumerate(label_counts)
        }

    return stats