#!/usr/bin/env python3
"""
ClinicalBERT Model Evaluation Script

This script provides comprehensive evaluation capabilities for ClinicalBERT models,
including both fine-tuned models and base models from HuggingFace.

Features:
- Load and evaluate any pretrained ClinicalBERT model
- Support for multiple data formats
- Comprehensive metrics and visualizations
- Model comparison capabilities
- Batch evaluation for large datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluation.evaluator import ClinicalBERTEvaluator
from src.utils.device_manager import setup_devices_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate ClinicalBERT models on readmission prediction tasks"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model or HuggingFace model name"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data (JSON format)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation configuration file"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Data doesn't contain labels (inference only)"
    )

    parser.add_argument(
        "--compare_models",
        type=str,
        nargs="+",
        help="List of model paths to compare"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/ssd-shared/yichen_models",
        help="Directory for model caching"
    )

    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate human-readable evaluation report"
    )

    return parser.parse_args()


@hydra.main(version_base=None, config_path="configs/evaluation", config_name="default")
def main_with_hydra(cfg: DictConfig) -> None:
    """Main function when using Hydra configuration"""
    logger.info("Starting ClinicalBERT model evaluation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup CUDA devices if specified
    device_manager = setup_devices_from_config(cfg) if hasattr(cfg, 'cuda_visible_devices') else None

    # Initialize evaluator
    evaluator = ClinicalBERTEvaluator(cache_dir=cfg.cache_dir, device_manager=device_manager)

    # Run evaluation
    try:
        if hasattr(cfg, 'compare_models') and cfg.compare_models:
            # Model comparison mode
            logger.info(f"Comparing {len(cfg.compare_models)} models")
            results = evaluator.compare_models(
                model_paths=cfg.compare_models,
                data_path=cfg.data_path,
                output_dir=cfg.output_dir,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                has_labels=not cfg.get('no_labels', False)
            )
            logger.info("Model comparison completed successfully")

        else:
            # Single model evaluation
            results = evaluator.evaluate_model(
                model_path=cfg.model_path,
                data_path=cfg.data_path,
                output_dir=cfg.output_dir,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                has_labels=not cfg.get('no_labels', False)
            )

            # Generate report if requested
            if cfg.get('generate_report', False):
                report_path = Path(cfg.output_dir) / 'evaluation_report.md'
                evaluator.generate_report(results, str(report_path))

            logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main_cli():
    """Main function for CLI usage"""
    args = parse_args()

    logger.info("Starting ClinicalBERT model evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")

    # Initialize evaluator
    evaluator = ClinicalBERTEvaluator(cache_dir=args.cache_dir)

    try:
        if args.compare_models:
            # Model comparison mode
            model_paths = [args.model_path] + args.compare_models
            logger.info(f"Comparing {len(model_paths)} models")

            results = evaluator.compare_models(
                model_paths=model_paths,
                data_path=args.data_path,
                output_dir=args.output_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                has_labels=not args.no_labels
            )
            logger.info("Model comparison completed successfully")

        else:
            # Single model evaluation
            results = evaluator.evaluate_model(
                model_path=args.model_path,
                data_path=args.data_path,
                output_dir=args.output_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                has_labels=not args.no_labels
            )

            # Generate report if requested
            if args.generate_report:
                report_path = Path(args.output_dir) / 'evaluation_report.md'
                evaluator.generate_report(results, str(report_path))

            logger.info("Model evaluation completed successfully")

            # Print summary to console
            if not args.no_labels:
                print("\n" + "="*50)
                print("EVALUATION SUMMARY")
                print("="*50)
                print(f"Model: {args.model_path}")
                print(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
                print(f"F1-Score (weighted): {results.get('f1_weighted', 'N/A'):.4f}")
                print(f"Precision (weighted): {results.get('precision_weighted', 'N/A'):.4f}")
                print(f"Recall (weighted): {results.get('recall_weighted', 'N/A'):.4f}")
                if 'auc_roc' in results:
                    print(f"AUC-ROC: {results['auc_roc']:.4f}")
                print("="*50)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we're being called with Hydra configuration
    if len(sys.argv) > 1 and any("hydra" in arg or "config" in arg for arg in sys.argv):
        main_with_hydra()
    else:
        main_cli()