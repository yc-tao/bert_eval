# Modularized ClinicalBERT Fine-tuning and Evaluation

This repository contains a modularized implementation for fine-tuning and evaluating ClinicalBERT models on hospital readmission prediction tasks.

## 🆕 New Features

### Modular Architecture
- **Clean separation of concerns** with organized modules
- **Reusable components** for data loading, model management, training, and evaluation
- **Flexible configuration system** using Hydra

### Pretrained Model Evaluation
- **Direct evaluation** of any pretrained ClinicalBERT model
- **Model comparison** capabilities for multiple models
- **Comprehensive metrics** with visualizations
- **Batch processing** for large datasets
- **Inference mode** for unlabeled data

## 📁 Project Structure

```
bert_eval/
├── src/                          # Modular source code
│   ├── data/                     # Data handling modules
│   │   ├── dataset.py           # Dataset classes
│   │   └── loader.py            # Data loading utilities
│   ├── models/                   # Model management
│   │   └── model_manager.py     # Model loading and caching
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Custom trainers and training args
│   ├── evaluation/               # Evaluation modules
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── evaluator.py         # Main evaluation orchestrator
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── config.yaml              # Main training config
│   ├── data/                    # Data configurations
│   ├── model/                   # Model configurations
│   ├── training/                # Training configurations
│   └── evaluation/              # Evaluation configurations
│       ├── default.yaml         # Default evaluation settings
│       ├── pretrained.yaml      # Pretrained model evaluation
│       ├── comparison.yaml      # Multi-model comparison
│       └── inference.yaml       # Inference without labels
├── train.py                     # Modular training script
├── evaluate.py                  # New evaluation script
├── finetune_clinical_bert.py    # Original monolithic script
├── test_modular.py              # Test suite for modular components
└── requirements.txt             # Updated dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Setup

```bash
# Test modular components
python test_modular.py

# Test data loading (original test)
python test_data_loading.py
```

### 3. Training (Modular)

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py --config-path configs --config-name config
```

### 4. Evaluate Pretrained Models

```bash
# Evaluate a HuggingFace model
python evaluate.py --model_path "medicalai/ClinicalBERT" \
                   --data_path "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json" \
                   --output_dir "./evaluation_results"

# Evaluate your fine-tuned model
python evaluate.py --model_path "/ssd-shared/yichen_models/clinical_bert_readmission" \
                   --data_path "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json" \
                   --output_dir "./evaluation_results" \
                   --generate_report

# Compare multiple models
python evaluate.py --model_path "medicalai/ClinicalBERT" \
                   --compare_models "emilyalsentzer/Bio_ClinicalBERT" "/ssd-shared/yichen_models/clinical_bert_readmission" \
                   --data_path "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json" \
                   --output_dir "./model_comparison"
```

## 📋 Evaluation Configurations

### Using Configuration Files

```bash
# Evaluate with default settings
python evaluate.py --config configs/evaluation/default.yaml

# Evaluate pretrained model
python evaluate.py --config configs/evaluation/pretrained.yaml

# Compare multiple models
python evaluate.py --config configs/evaluation/comparison.yaml

# Inference without labels
python evaluate.py --config configs/evaluation/inference.yaml
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Model path or HuggingFace name | Required |
| `--data_path` | Path to evaluation data | Required |
| `--output_dir` | Results output directory | `./evaluation_results` |
| `--max_length` | Maximum sequence length | `256` |
| `--batch_size` | Evaluation batch size | `32` |
| `--no_labels` | Data has no labels (inference) | `False` |
| `--compare_models` | Additional models to compare | `None` |
| `--generate_report` | Generate human-readable report | `False` |

## 📊 Evaluation Output

### Single Model Evaluation

```
evaluation_results/
├── evaluation_results.json      # Comprehensive metrics
├── evaluation_report.md         # Human-readable report
├── confusion_matrix.png         # Confusion matrix plot
├── roc_curve.png               # ROC curve
└── precision_recall_curve.png  # Precision-recall curve
```

### Model Comparison

```
model_comparison/
├── model_comparison.json        # Comparison summary
├── model_1_*/                  # Individual model results
├── model_2_*/
└── model_3_*/
```

### Inference Mode

```
inference_results/
└── predictions.json             # Model predictions and probabilities
```

## 🔧 API Usage

### Programmatic Evaluation

```python
from src.evaluation.evaluator import ClinicalBERTEvaluator

# Initialize evaluator
evaluator = ClinicalBERTEvaluator()

# Evaluate single model
results = evaluator.evaluate_model(
    model_path="medicalai/ClinicalBERT",
    data_path="/path/to/data.json",
    output_dir="./results"
)

# Compare models
comparison = evaluator.compare_models(
    model_paths=["model1", "model2", "model3"],
    data_path="/path/to/data.json",
    output_dir="./comparison"
)
```

### Custom Model Loading

```python
from src.models.model_manager import ModelManager

# Initialize model manager
manager = ModelManager(cache_dir="/ssd-shared/yichen_models")

# Load model and tokenizer
model, tokenizer = manager.setup_model_for_evaluation("your_model_path")

# Get model information
info = manager.get_model_info("medicalai/ClinicalBERT")
```

## 📈 Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **AUC-ROC**: Area under ROC curve (binary classification)
- **Confusion Matrix**: Detailed classification breakdown

### Advanced Metrics
- **ROC Curves**: True positive vs false positive rates
- **Precision-Recall Curves**: Precision vs recall trade-offs
- **Per-class Statistics**: Individual class performance
- **Support**: Number of samples per class

### Visualizations
- Confusion matrix heatmaps
- ROC curve plots
- Precision-recall curve plots
- Model comparison charts (when comparing multiple models)

## 🔄 Migration from Original Script

The original `finetune_clinical_bert.py` is still available for compatibility. Key differences:

| Feature | Original | Modular |
|---------|----------|---------|
| **Architecture** | Monolithic | Modular components |
| **Evaluation** | Training only | Separate evaluation script |
| **Model Loading** | Hardcoded fallback | Flexible model manager |
| **Configuration** | Single config | Multiple specialized configs |
| **Testing** | Basic data test | Comprehensive test suite |
| **Extensibility** | Limited | Highly extensible |

## 🧪 Testing

### Run All Tests

```bash
python test_modular.py
```

### Individual Component Testing

```python
# Test specific components
from test_modular import test_imports, test_model_manager, test_data_loading

test_imports()
test_model_manager()
test_data_loading()
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing Visualization**: Install optional plotting dependencies
   ```bash
   pip install matplotlib seaborn
   ```

3. **Model Loading Failures**: Check internet connection and HuggingFace access

4. **Memory Issues**: Reduce batch size in configuration
   ```yaml
   batch_size: 16  # Reduce from 32
   ```

5. **Configuration Errors**: Validate YAML syntax and file paths

### Debug Mode

Enable detailed logging:

```bash
python evaluate.py --model_path "your_model" --data_path "your_data" --output_dir "results" --verbose
```

## 🤝 Contributing

The modular architecture makes it easy to extend functionality:

1. **Add new evaluation metrics** in `src/evaluation/metrics.py`
2. **Implement new model types** in `src/models/model_manager.py`
3. **Create custom datasets** in `src/data/dataset.py`
4. **Add visualization types** in `src/evaluation/evaluator.py`

## 📝 Configuration Examples

### Custom Evaluation Config

```yaml
# configs/evaluation/custom.yaml
model_path: "your/custom/model"
data_path: "/path/to/your/data.json"
max_length: 512
batch_size: 16
output_dir: "./custom_evaluation"
generate_report: true
generate_plots: true
```

### Model Comparison Config

```yaml
# configs/evaluation/my_comparison.yaml
model_path: "medicalai/ClinicalBERT"
compare_models:
  - "emilyalsentzer/Bio_ClinicalBERT"
  - "/path/to/your/finetuned/model"
data_path: "/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json"
output_dir: "./my_model_comparison"
max_length: 256
batch_size: 32
```

## 📄 License

This project maintains the same license as the original ClinicalBERT fine-tuning implementation.