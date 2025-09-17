# ClinicalBERT Fine-tuning for Readmission Prediction

This repository contains a script for fine-tuning ClinicalBERT models from HuggingFace to predict hospital readmissions using MIMIC-IV discharge summaries.

## Features

- **Automated Model Caching**: Downloads and caches ClinicalBERT models to `/ssd-shared/yichen_models/`
- **Left Truncation Strategy**: Handles long clinical texts by keeping the most recent information (discharge details)
- **Class Imbalance Handling**: Uses weighted loss to handle imbalanced readmission data
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and AUC-ROC metrics
- **GPU Optimization**: Supports mixed precision training and automatic GPU detection

## Dataset

The script works with MIMIC-IV readmission data located at:
```
/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json
```

**Dataset Statistics:**
- Total samples: 25,160 discharge summaries
- Label distribution: 19,361 (no readmission) vs 5,799 (readmission) - 77%/23% split
- Text length: 1,136 - 58,225 characters (avg: 10,692 chars)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Test the setup:
```bash
python test_data_loading.py
```

## Usage

### Basic Usage

Run with default settings:
```bash
python finetune_clinical_bert.py
```

### Advanced Usage

Customize training parameters:
```bash
# Use different model
python finetune_clinical_bert.py --model "emilyalsentzer/Bio_ClinicalBERT"

# Adjust training epochs
python finetune_clinical_bert.py --epochs 5

# Change batch size
python finetune_clinical_bert.py --batch_size 32
```

### Model Selection

The script supports two ClinicalBERT variants:

1. **medicalai/ClinicalBERT** (default)
   - Max sequence length: 256 tokens
   - Primary clinical BERT implementation

2. **emilyalsentzer/Bio_ClinicalBERT** (fallback)
   - Max sequence length: 128 tokens
   - Alternative clinical BERT with optimized tokenization

## Configuration

Key parameters can be modified in the `Config` class:

```python
@dataclass
class Config:
    # Model settings
    model_name: str = "medicalai/ClinicalBERT"
    max_length: int = 256

    # Training parameters
    train_batch_size: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: int = 3

    # Paths
    data_path: str = "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json"
    model_cache_dir: str = "/ssd-shared/yichen_models"
    output_dir: str = "/ssd-shared/yichen_models/clinical_bert_readmission"
```

## Output

The script will save:

1. **Fine-tuned model**: Complete model and tokenizer in `/ssd-shared/yichen_models/clinical_bert_readmission/`
2. **Training logs**: Training progress and validation metrics
3. **Evaluation results**: Detailed performance metrics in `evaluation_results.json`

### Sample Output Structure
```
/ssd-shared/yichen_models/clinical_bert_readmission/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── evaluation_results.json
└── logs/
    └── training_logs.txt
```

## Training Process

1. **Data Loading**: Loads and analyzes the MIMIC-IV dataset
2. **Model Setup**: Downloads and caches ClinicalBERT from HuggingFace
3. **Data Preprocessing**: Applies left truncation and tokenization
4. **Training**: Fine-tunes with weighted loss for class imbalance
5. **Evaluation**: Comprehensive metrics on validation set
6. **Model Saving**: Saves the best performing model

## Performance Considerations

- **Memory**: Requires ~4-8GB GPU memory for training
- **Time**: Approximately 2-3 hours on modern GPU (depends on batch size and epochs)
- **Storage**: Model cache requires ~500MB, fine-tuned model ~500MB

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `train_batch_size` in the config
2. **Slow Training**: Enable `fp16=True` for mixed precision training
3. **Model Download Fails**: Check internet connection and HuggingFace access

### Environment Check

Run the test script to verify setup:
```bash
python test_data_loading.py
```

This will check:
- Data file accessibility and format
- Model cache directory permissions
- Basic statistics about the dataset

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- ~10GB free disk space for models and cache

See `requirements.txt` for complete dependency list. 
