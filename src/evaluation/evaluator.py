#!/usr/bin/env python3
"""
Main evaluation orchestrator for ClinicalBERT models
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..models.model_manager import ModelManager
from ..data.loader import load_evaluation_data, get_data_statistics
from ..data.dataset import EvaluationDataset
from .metrics import ModelEvaluator
from ..utils.device_manager import DeviceManager

logger = logging.getLogger(__name__)


class ClinicalBERTEvaluator:
    """Main orchestrator for evaluating ClinicalBERT models"""

    def __init__(self, cache_dir: str = "/ssd-shared/yichen_models", device_manager: DeviceManager = None):
        self.model_manager = ModelManager(cache_dir)
        self.device_manager = device_manager or DeviceManager()
        self.results = {}

    def evaluate_model(self, model_path: str, data_path: str,
                      output_dir: str, max_length: int = 256,
                      batch_size: int = 32, has_labels: bool = True) -> Dict:
        """
        Evaluate a single model on a dataset

        Args:
            model_path: Path to model or HuggingFace model name
            data_path: Path to evaluation data
            output_dir: Directory to save results
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            has_labels: Whether the data has labels

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation of model: {model_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Output directory: {output_dir}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer with device management
        try:
            # Determine device for evaluation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = self.model_manager.setup_model_for_evaluation(model_path, device=device)
            logger.info("Model loaded successfully")

            # Log device information
            if self.device_manager:
                self.device_manager.log_device_setup()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Load evaluation data
        try:
            texts, labels = load_evaluation_data(data_path, has_labels)
            logger.info(f"Loaded {len(texts)} samples for evaluation")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        # Get data statistics
        data_stats = get_data_statistics(texts, labels)
        logger.info(f"Data statistics: {data_stats}")

        # Create evaluator
        evaluator = ModelEvaluator(model, tokenizer)

        # Run evaluation
        if has_labels and labels is not None:
            logger.info("Running evaluation with labels")

            if batch_size > 1:
                # Batch evaluation using dataset
                dataset = EvaluationDataset(texts, tokenizer, max_length, labels)
                results = evaluator.evaluate_batch(dataset, batch_size)
            else:
                # Single sample evaluation
                results = evaluator.evaluate(texts, labels, max_length)

            # Add model and data info
            results['model_info'] = {
                'model_path': model_path,
                'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
            }
            results['data_info'] = {
                'data_path': data_path,
                'data_statistics': data_stats
            }
            results['evaluation_params'] = {
                'max_length': max_length,
                'batch_size': batch_size
            }

            # Save results
            results_path = Path(output_dir) / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_path}")

            # Generate visualizations if binary classification
            if len(np.unique(labels)) == 2:
                self._generate_visualizations(results, output_dir)

        else:
            logger.info("Running inference without labels")
            predictions, probabilities = evaluator.predict(texts, max_length)

            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'model_info': {
                    'model_path': model_path,
                    'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
                },
                'data_info': {
                    'data_path': data_path,
                    'data_statistics': data_stats
                },
                'evaluation_params': {
                    'max_length': max_length,
                    'batch_size': batch_size
                }
            }

            # Save predictions
            results_path = Path(output_dir) / 'predictions.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Predictions saved to {results_path}")

        self.results[model_path] = results
        return results

    def compare_models(self, model_paths: List[str], data_path: str,
                      output_dir: str, **kwargs) -> Dict:
        """
        Compare multiple models on the same dataset

        Args:
            model_paths: List of model paths or names
            data_path: Path to evaluation data
            output_dir: Directory to save comparison results
            **kwargs: Additional arguments for evaluate_model

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(model_paths)} models")

        comparison_results = {}
        individual_results = {}

        for i, model_path in enumerate(model_paths):
            logger.info(f"Evaluating model {i+1}/{len(model_paths)}: {model_path}")

            model_output_dir = Path(output_dir) / f"model_{i+1}_{Path(model_path).name}"
            try:
                results = self.evaluate_model(model_path, data_path, str(model_output_dir), **kwargs)
                individual_results[model_path] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
                individual_results[model_path] = {"error": str(e)}

        # Create comparison summary
        comparison_results['individual_results'] = individual_results
        comparison_results['summary'] = self._create_comparison_summary(individual_results)

        # Save comparison results
        comparison_path = Path(output_dir) / 'model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)

        logger.info(f"Comparison results saved to {comparison_path}")
        return comparison_results

    def _create_comparison_summary(self, individual_results: Dict) -> Dict:
        """Create a summary comparison of multiple models"""
        summary = {
            'metrics_comparison': {},
            'best_model': {},
            'model_rankings': {}
        }

        # Extract key metrics for comparison
        metrics_to_compare = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        if any('auc_roc' in results for results in individual_results.values()
               if isinstance(results, dict) and 'error' not in results):
            metrics_to_compare.append('auc_roc')

        for metric in metrics_to_compare:
            metric_values = {}
            for model_path, results in individual_results.items():
                if isinstance(results, dict) and 'error' not in results and metric in results:
                    metric_values[model_path] = results[metric]

            if metric_values:
                summary['metrics_comparison'][metric] = metric_values
                # Find best model for this metric
                best_model = max(metric_values.items(), key=lambda x: x[1])
                summary['best_model'][metric] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }

        return summary

    def _generate_visualizations(self, results: Dict, output_dir: str):
        """Generate visualization plots for binary classification"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for visualizations")
            return

        try:

            output_path = Path(output_dir)

            # Confusion Matrix
            if 'confusion_matrix' in results:
                plt.figure(figsize=(8, 6))
                cm = np.array(results['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['No Readmission', 'Readmission'],
                           yticklabels=['No Readmission', 'Readmission'])
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()

            # ROC Curve
            if 'roc_curve' in results:
                plt.figure(figsize=(8, 6))
                roc_data = results['roc_curve']
                plt.plot(roc_data['fpr'], roc_data['tpr'],
                        label=f"ROC Curve (AUC = {results.get('auc_roc', 0):.3f})")
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Precision-Recall Curve
            if 'pr_curve' in results:
                plt.figure(figsize=(8, 6))
                pr_data = results['pr_curve']
                plt.plot(pr_data['recall'], pr_data['precision'])
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
                plt.close()

            logger.info(f"Visualizations saved to {output_dir}")

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def generate_report(self, results: Dict, output_path: str):
        """Generate a human-readable evaluation report"""
        report_lines = []
        report_lines.append("# ClinicalBERT Evaluation Report\n")

        # Model information
        if 'model_info' in results:
            model_info = results['model_info']
            report_lines.append("## Model Information")
            report_lines.append(f"- **Model Path**: {model_info.get('model_path', 'N/A')}")
            if 'model_config' in model_info:
                config = model_info['model_config']
                report_lines.append(f"- **Model Type**: {config.get('model_type', 'N/A')}")
                report_lines.append(f"- **Hidden Size**: {config.get('hidden_size', 'N/A')}")
                report_lines.append(f"- **Number of Labels**: {config.get('num_labels', 'N/A')}")
            report_lines.append("")

        # Data information
        if 'data_info' in results:
            data_info = results['data_info']
            report_lines.append("## Dataset Information")
            report_lines.append(f"- **Data Path**: {data_info.get('data_path', 'N/A')}")
            if 'data_statistics' in data_info:
                stats = data_info['data_statistics']
                report_lines.append(f"- **Number of Samples**: {stats.get('num_samples', 'N/A')}")
                if 'label_distribution' in stats:
                    dist = stats['label_distribution']
                    report_lines.append(f"- **Label Distribution**: {dist}")
            report_lines.append("")

        # Performance metrics
        report_lines.append("## Performance Metrics")
        report_lines.append(f"- **Accuracy**: {results.get('accuracy', 'N/A'):.4f}")
        report_lines.append(f"- **Weighted F1-Score**: {results.get('f1_weighted', 'N/A'):.4f}")
        report_lines.append(f"- **Weighted Precision**: {results.get('precision_weighted', 'N/A'):.4f}")
        report_lines.append(f"- **Weighted Recall**: {results.get('recall_weighted', 'N/A'):.4f}")

        if 'auc_roc' in results:
            report_lines.append(f"- **AUC-ROC**: {results['auc_roc']:.4f}")

        report_lines.append("")

        # Per-class metrics
        if 'precision_per_class' in results:
            report_lines.append("## Per-Class Metrics")
            classes = ['No Readmission', 'Readmission'] if len(results['precision_per_class']) == 2 else [f'Class {i}' for i in range(len(results['precision_per_class']))]

            for i, class_name in enumerate(classes):
                if i < len(results['precision_per_class']):
                    report_lines.append(f"### {class_name}")
                    report_lines.append(f"- **Precision**: {results['precision_per_class'][i]:.4f}")
                    report_lines.append(f"- **Recall**: {results['recall_per_class'][i]:.4f}")
                    report_lines.append(f"- **F1-Score**: {results['f1_per_class'][i]:.4f}")
                    report_lines.append(f"- **Support**: {results['support_per_class'][i]}")
                    report_lines.append("")

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Evaluation report saved to {output_path}")