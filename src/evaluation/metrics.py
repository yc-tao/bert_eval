#!/usr/bin/env python3
"""
Evaluation metrics for ClinicalBERT models
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)


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


class ModelEvaluator:
    """Comprehensive model evaluation class"""

    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], max_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and probabilities for a list of texts

        Args:
            texts: List of text samples
            max_length: Maximum sequence length for tokenization

        Returns:
            Tuple of (predictions, probabilities)
        """
        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize the text
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )

                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get probabilities and predictions
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                prediction = np.argmax(probabilities)

                all_predictions.append(prediction)
                all_probabilities.append(probabilities)

        return np.array(all_predictions), np.array(all_probabilities)

    def evaluate(self, texts: List[str], labels: List[int], max_length: int = 256) -> Dict:
        """
        Comprehensive evaluation of the model

        Args:
            texts: List of text samples
            labels: List of true labels
            max_length: Maximum sequence length

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        predictions, probabilities = self.predict(texts, max_length)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        results = {
            'accuracy': float(accuracy),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
        }

        # Binary classification specific metrics
        if len(np.unique(labels)) == 2:
            # AUC-ROC
            auc_roc = roc_auc_score(labels, probabilities[:, 1])
            results['auc_roc'] = float(auc_roc)

            # ROC curve data
            fpr, tpr, roc_thresholds = roc_curve(labels, probabilities[:, 1])
            results['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }

            # Precision-Recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                labels, probabilities[:, 1]
            )
            results['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }

        # Classification report
        target_names = [f'Class_{i}' for i in range(len(np.unique(labels)))]
        if len(target_names) == 2:
            target_names = ['No Readmission', 'Readmission']

        results['classification_report'] = classification_report(
            labels, predictions, target_names=target_names, output_dict=True
        )

        return results

    def evaluate_batch(self, dataset, batch_size: int = 32) -> Dict:
        """
        Evaluate model on a dataset in batches

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation results
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_labels = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)

                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities)

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Use the same evaluation logic as evaluate()
        texts = None  # We don't have access to original texts in batch mode
        return self._compute_metrics_from_arrays(all_labels, all_predictions, all_probabilities)

    def _compute_metrics_from_arrays(self, labels: np.ndarray, predictions: np.ndarray,
                                   probabilities: np.ndarray) -> Dict:
        """Helper method to compute metrics from arrays"""
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        results = {
            'accuracy': float(accuracy),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
        }

        # Binary classification specific metrics
        if len(np.unique(labels)) == 2 and probabilities.shape[1] >= 2:
            # AUC-ROC
            auc_roc = roc_auc_score(labels, probabilities[:, 1])
            results['auc_roc'] = float(auc_roc)

            # ROC curve data
            fpr, tpr, roc_thresholds = roc_curve(labels, probabilities[:, 1])
            results['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }

            # Precision-Recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                labels, probabilities[:, 1]
            )
            results['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }

        return results