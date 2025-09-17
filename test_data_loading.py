#!/usr/bin/env python3
"""
Simple test script to validate data loading without ML dependencies
"""

import json
import os

def test_data_loading():
    """Test loading and basic analysis of the dataset"""
    data_path = "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json"

    print(f"Testing data loading from: {data_path}")

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return False

    # Load and analyze data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"Successfully loaded {len(data)} samples")

        # Analyze first sample
        if data:
            first_sample = data[0]
            print(f"Sample keys: {list(first_sample.keys())}")
            print(f"Text length: {len(first_sample['text'])} characters")
            print(f"Label: {first_sample['label']}")

        # Label distribution
        labels = [item['label'] for item in data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"Label distribution: {label_counts}")

        # Text length statistics
        text_lengths = [len(item['text']) for item in data]
        print(f"Text length stats:")
        print(f"  Min: {min(text_lengths)} chars")
        print(f"  Max: {max(text_lengths)} chars")
        print(f"  Average: {sum(text_lengths)/len(text_lengths):.0f} chars")

        return True

    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False

def test_model_cache_dir():
    """Test model cache directory setup"""
    cache_dir = "/ssd-shared/yichen_models"
    print(f"\nTesting model cache directory: {cache_dir}")

    if os.path.exists(cache_dir):
        print(f"Cache directory exists")
        print(f"Directory contents: {os.listdir(cache_dir)}")

        # Test write permissions
        try:
            test_file = os.path.join(cache_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("Write permissions: OK")
            return True
        except Exception as e:
            print(f"Write permission error: {e}")
            return False
    else:
        print(f"Cache directory does not exist")
        return False

if __name__ == "__main__":
    print("=== Testing ClinicalBERT Fine-tuning Setup ===\n")

    success = True
    success &= test_data_loading()
    success &= test_model_cache_dir()

    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed! Ready for fine-tuning.")
        print("\nTo run the fine-tuning script, first install dependencies:")
        print("pip install -r requirements.txt")
        print("\nThen run:")
        print("python finetune_clinical_bert.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")