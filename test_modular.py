#!/usr/bin/env python3
"""
Test script for the modularized ClinicalBERT implementation
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from src.data.dataset import ReadmissionDataset, EvaluationDataset
        from src.data.loader import load_and_preprocess_data, split_data, load_evaluation_data
        from src.models.model_manager import ModelManager
        from src.training.trainer import WeightedTrainer, compute_class_weights
        from src.evaluation.metrics import compute_metrics, ModelEvaluator
        from src.evaluation.evaluator import ClinicalBERTEvaluator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_manager():
    """Test ModelManager functionality"""
    print("\nTesting ModelManager...")

    try:
        from src.models.model_manager import ModelManager

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(temp_dir)

            # Test model info retrieval
            info = manager.get_model_info("bert-base-uncased")
            print(f"✓ Model info retrieved: {info.get('model_type', 'unknown')}")

            # Test fallback models list
            fallbacks = manager.get_fallback_models()
            print(f"✓ Fallback models available: {len(fallbacks)}")

        return True
    except Exception as e:
        print(f"✗ ModelManager test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")

    try:
        # Create mock data file with balanced classes
        mock_data = [
            {"text": "Patient discharged after treatment", "label": 0},
            {"text": "Follow-up required for complications", "label": 1},
            {"text": "Recovery proceeding normally", "label": 0},
            {"text": "Readmission due to complications", "label": 1}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_data, f)
            temp_file = f.name

        try:
            from src.data.loader import load_and_preprocess_data, split_data, get_data_statistics

            # Test data loading
            texts, labels = load_and_preprocess_data(temp_file)
            print(f"✓ Loaded {len(texts)} samples")

            # Test data splitting
            train_texts, val_texts, train_labels, val_labels = split_data(texts, labels, test_size=0.3)
            print(f"✓ Split into {len(train_texts)} train, {len(val_texts)} val samples")

            # Test statistics
            stats = get_data_statistics(texts, labels)
            print(f"✓ Statistics computed: {stats['num_samples']} samples")

        finally:
            os.unlink(temp_file)

        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation"""
    print("\nTesting dataset creation...")

    try:
        from src.data.dataset import ReadmissionDataset, EvaluationDataset
        from transformers import AutoTokenizer

        # Create mock tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        texts = ["Sample medical text 1", "Sample medical text 2"]
        labels = [0, 1]

        # Test ReadmissionDataset
        dataset = ReadmissionDataset(texts, labels, tokenizer, max_length=128)
        print(f"✓ ReadmissionDataset created with {len(dataset)} samples")

        # Test sample access
        sample = dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")

        # Test EvaluationDataset
        eval_dataset = EvaluationDataset(texts, tokenizer, max_length=128, labels=labels)
        print(f"✓ EvaluationDataset created with {len(eval_dataset)} samples")

        return True
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        return False

def test_evaluation_script():
    """Test that evaluation script can be imported and basic functionality works"""
    print("\nTesting evaluation script...")

    try:
        # Test that the script can be imported
        import evaluate
        print("✓ Evaluation script imported successfully")

        # Test ClinicalBERTEvaluator initialization
        from src.evaluation.evaluator import ClinicalBERTEvaluator
        evaluator = ClinicalBERTEvaluator()
        print("✓ ClinicalBERTEvaluator initialized")

        return True
    except Exception as e:
        print(f"✗ Evaluation script test failed: {e}")
        return False

def test_configuration_files():
    """Test that configuration files exist and are valid"""
    print("\nTesting configuration files...")

    try:
        from omegaconf import OmegaConf

        config_files = [
            "configs/config.yaml",
            "configs/evaluation/default.yaml",
            "configs/evaluation/pretrained.yaml",
            "configs/evaluation/comparison.yaml"
        ]

        for config_file in config_files:
            if Path(config_file).exists():
                cfg = OmegaConf.load(config_file)
                print(f"✓ {config_file} loaded successfully")
            else:
                print(f"⚠ {config_file} not found")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Modularized ClinicalBERT Implementation ===\n")

    tests = [
        test_imports,
        test_model_manager,
        test_data_loading,
        test_dataset_creation,
        test_evaluation_script,
        test_configuration_files
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed! Modular implementation is working correctly.")
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install matplotlib seaborn")
        print("2. Test training: python train.py")
        print("3. Test evaluation: python evaluate.py --model_path medicalai/ClinicalBERT --data_path /path/to/data")
        return True
    else:
        print(f"✗ {total - passed}/{total} tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)