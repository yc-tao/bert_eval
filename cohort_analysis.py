#!/usr/bin/env python3
"""
Cohort Analysis Script for MIMIC-IV Readmission Data

This script performs cohort analysis comparing:
1. Tuning data from configs/data/default.yaml
2. Training split from /ssd-shared/ryans_output/mimic_iv_readmission_train.json

It also analyzes data distribution for train/val splits using seed 42 and split size 0.2.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import os

def load_json_data(file_path):
    """Load JSON data from file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data

def extract_demographics(text):
    """Extract basic demographics from discharge summary text."""
    demographics = {}

    # Extract sex
    if "Sex:   F" in text or "Sex: F" in text:
        demographics['sex'] = 'F'
    elif "Sex:   M" in text or "Sex: M" in text:
        demographics['sex'] = 'M'
    else:
        demographics['sex'] = 'Unknown'

    # Extract service
    service_line = None
    for line in text.split('\n'):
        if line.strip().startswith('Service:'):
            service_line = line.strip()
            break

    if service_line:
        demographics['service'] = service_line.replace('Service:', '').strip()
    else:
        demographics['service'] = 'Unknown'

    return demographics

def analyze_cohort_characteristics(data, dataset_name, service_sample_size=5000):
    """Analyze cohort characteristics."""
    print(f"\n=== {dataset_name} Cohort Analysis ===")

    # Basic statistics
    total_records = len(data)
    print(f"Total records: {total_records:,}")

    # Label distribution
    labels = [record['label'] for record in data]
    label_counts = Counter(labels)
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_records) * 100
        print(f"  Label {label}: {count:,} ({percentage:.2f}%)")

    # Subject ID analysis
    subject_ids = [record['subject_id'] for record in data]
    unique_subjects = len(set(subject_ids))
    print(f"Unique subjects: {unique_subjects:,}")
    print(f"Records per subject (avg): {total_records/unique_subjects:.2f}")

    # Admission ID analysis
    hadm_ids = [record['hadm_id'] for record in data if 'hadm_id' in record]
    unique_admissions = len(set(hadm_ids))
    print(f"Unique admissions: {unique_admissions:,}")

    # Note type analysis (if available)
    note_types = [record.get('note_type', 'Unknown') for record in data]
    note_type_counts = Counter(note_types)
    print(f"Note types:")
    for note_type, count in note_type_counts.most_common(10):
        percentage = (count / total_records) * 100
        print(f"  {note_type}: {count:,} ({percentage:.2f}%)")

    # Demographics analysis with consistent sampling
    # For service analysis, use larger sample size
    service_sample_size = min(service_sample_size, len(data))
    service_sample_data = np.random.choice(data, service_sample_size, replace=False)

    # For sex analysis, use smaller sample for performance
    sex_sample_size = min(1000, len(data))
    sex_sample_data = np.random.choice(data, sex_sample_size, replace=False)

    service_demographics = []
    for record in service_sample_data:
        demo = extract_demographics(record['text'])
        service_demographics.append(demo)

    sex_demographics = []
    for record in sex_sample_data:
        demo = extract_demographics(record['text'])
        sex_demographics.append(demo)

    # Sex distribution
    sex_counts = Counter([d['sex'] for d in sex_demographics])
    print(f"Sex distribution (sample of {sex_sample_size}):")
    for sex, count in sex_counts.items():
        percentage = (count / sex_sample_size) * 100
        print(f"  {sex}: {count} ({percentage:.2f}%)")

    # Service distribution
    service_counts = Counter([d['service'] for d in service_demographics])
    print(f"Top services (sample of {service_sample_size}):")
    for service, count in service_counts.most_common(10):
        percentage = (count / service_sample_size) * 100
        print(f"  {service}: {count} ({percentage:.2f}%)")

    return {
        'total_records': total_records,
        'unique_subjects': unique_subjects,
        'unique_admissions': unique_admissions,
        'label_distribution': dict(label_counts),
        'note_type_distribution': dict(note_type_counts),
        'sex_distribution': dict(sex_counts),
        'service_distribution': dict(service_counts),
        'service_sample_size': service_sample_size
    }

def compare_cohorts(tuning_train_stats, tuning_val_stats, training_stats):
    """Compare characteristics between all three cohorts."""
    print(f"\n=== Cohort Comparison ===")

    print(f"Dataset sizes:")
    print(f"  Tuning Train (80%): {tuning_train_stats['total_records']:,} records")
    print(f"  Tuning Val (20%): {tuning_val_stats['total_records']:,} records")
    print(f"  External Training: {training_stats['total_records']:,} records")
    print(f"  Ratio (external/tuning_total): {training_stats['total_records']/(tuning_train_stats['total_records'] + tuning_val_stats['total_records']):.2f}")

    print(f"\nUnique subjects:")
    print(f"  Tuning Train: {tuning_train_stats['unique_subjects']:,}")
    print(f"  Tuning Val: {tuning_val_stats['unique_subjects']:,}")
    print(f"  External Training: {training_stats['unique_subjects']:,}")

    print(f"\nLabel distribution comparison:")
    for label in [0, 1]:
        tuning_train_pct = (tuning_train_stats['label_distribution'].get(label, 0) / tuning_train_stats['total_records']) * 100
        tuning_val_pct = (tuning_val_stats['label_distribution'].get(label, 0) / tuning_val_stats['total_records']) * 100
        training_pct = (training_stats['label_distribution'].get(label, 0) / training_stats['total_records']) * 100
        print(f"  Label {label}:")
        print(f"    Tuning Train: {tuning_train_pct:.2f}%")
        print(f"    Tuning Val: {tuning_val_pct:.2f}%")
        print(f"    External Training: {training_pct:.2f}%")

def analyze_train_val_split(data, test_size=0.2, random_state=42):
    """Analyze train/validation split characteristics and return split datasets."""
    print(f"\n=== Train/Validation Split Analysis ===")
    print(f"Split configuration: test_size={test_size}, random_state={random_state}")

    # Prepare data for splitting
    indices = list(range(len(data)))
    y = [record['label'] for record in data]

    # Perform stratified split on indices
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create split datasets
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    print(f"\nSplit sizes:")
    print(f"  Training: {len(train_data):,} records ({(1-test_size)*100:.0f}%)")
    print(f"  Validation: {len(val_data):,} records ({test_size*100:.0f}%)")

    # Analyze label distributions
    train_labels = [record['label'] for record in train_data]
    val_labels = [record['label'] for record in val_data]
    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)

    print(f"\nLabel distribution in training split:")
    for label, count in sorted(train_label_counts.items()):
        percentage = (count / len(train_data)) * 100
        print(f"  Label {label}: {count:,} ({percentage:.2f}%)")

    print(f"\nLabel distribution in validation split:")
    for label, count in sorted(val_label_counts.items()):
        percentage = (count / len(val_data)) * 100
        print(f"  Label {label}: {count:,} ({percentage:.2f}%)")

    # Subject overlap analysis
    train_subjects = set([record['subject_id'] for record in train_data])
    val_subjects = set([record['subject_id'] for record in val_data])
    overlap_subjects = train_subjects.intersection(val_subjects)

    print(f"\nSubject analysis:")
    print(f"  Unique subjects in train: {len(train_subjects):,}")
    print(f"  Unique subjects in val: {len(val_subjects):,}")
    print(f"  Overlapping subjects: {len(overlap_subjects):,}")
    print(f"  Subject leakage rate: {(len(overlap_subjects)/len(val_subjects))*100:.2f}%")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'train_labels': dict(train_label_counts),
        'val_labels': dict(val_label_counts),
        'train_subjects': len(train_subjects),
        'val_subjects': len(val_subjects),
        'overlapping_subjects': len(overlap_subjects)
    }

def create_visualizations(tuning_train_stats, tuning_val_stats, training_stats, split_stats, output_dir='outputs/figs'):
    """Create visualization plots for all three datasets."""
    print(f"\n=== Creating Visualizations ===")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set color palette
    colors = ['#500000', '#3E3E3E', '#D6D3C4']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MIMIC-IV Readmission Data Cohort Analysis', fontsize=18, fontweight='bold')

    # Plot 1: Dataset sizes comparison (all three datasets)
    ax1 = axes[0, 0]
    datasets = ['Tuning\nTrain', 'Tuning\nVal', 'External\nTraining']
    sizes = [tuning_train_stats['total_records'], tuning_val_stats['total_records'], training_stats['total_records']]
    bars = ax1.bar(datasets, sizes, color=colors, edgecolor='white', linewidth=1.2)
    ax1.set_title('Dataset Sizes Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{size:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Label distribution comparison (all three datasets)
    ax2 = axes[0, 1]
    labels = [0, 1]
    tuning_train_pcts = [(tuning_train_stats['label_distribution'].get(label, 0) / tuning_train_stats['total_records']) * 100 for label in labels]
    tuning_val_pcts = [(tuning_val_stats['label_distribution'].get(label, 0) / tuning_val_stats['total_records']) * 100 for label in labels]
    training_pcts = [(training_stats['label_distribution'].get(label, 0) / training_stats['total_records']) * 100 for label in labels]

    x = np.arange(len(labels))
    width = 0.25
    ax2.bar(x - width, tuning_train_pcts, width, label='Tuning Train', color=colors[0], edgecolor='white', linewidth=1.2)
    ax2.bar(x, tuning_val_pcts, width, label='Tuning Val', color=colors[1], edgecolor='white', linewidth=1.2)
    ax2.bar(x + width, training_pcts, width, label='External Training', color=colors[2], edgecolor='white', linewidth=1.2)
    ax2.set_title('Label Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xlabel('Label', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Label {label}' for label in labels], fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Plot 3: Service distribution comparison (top services from all datasets)
    ax3 = axes[0, 2]

    # Get top services across all datasets
    all_services = set()
    all_services.update(tuning_train_stats['service_distribution'].keys())
    all_services.update(tuning_val_stats['service_distribution'].keys())
    all_services.update(training_stats['service_distribution'].keys())

    # Calculate combined counts and get top 8
    service_totals = {}
    for service in all_services:
        total = (tuning_train_stats['service_distribution'].get(service, 0) +
                tuning_val_stats['service_distribution'].get(service, 0) +
                training_stats['service_distribution'].get(service, 0))
        service_totals[service] = total

    top_services = sorted(service_totals.items(), key=lambda x: x[1], reverse=True)[:8]
    top_service_names = [s[0] for s in top_services]

    # Get counts for each dataset
    tuning_train_counts = [tuning_train_stats['service_distribution'].get(service, 0) for service in top_service_names]
    tuning_val_counts = [tuning_val_stats['service_distribution'].get(service, 0) for service in top_service_names]
    training_counts = [training_stats['service_distribution'].get(service, 0) for service in top_service_names]

    y_pos = np.arange(len(top_service_names))
    bar_height = 0.25

    ax3.barh(y_pos - bar_height, tuning_train_counts, bar_height, label='Tuning Train', color=colors[0], edgecolor='white', linewidth=1)
    ax3.barh(y_pos, tuning_val_counts, bar_height, label='Tuning Val', color=colors[1], edgecolor='white', linewidth=1)
    ax3.barh(y_pos + bar_height, training_counts, bar_height, label='External Training', color=colors[2], edgecolor='white', linewidth=1)

    ax3.set_title('Top Services Comparison (5k sample each)', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Count', fontsize=12)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_service_names, fontsize=10)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Plot 4: Subject overlap in train/val split
    ax4 = axes[1, 0]
    overlap_data = [
        split_stats['train_subjects'] - split_stats['overlapping_subjects'],
        split_stats['val_subjects'] - split_stats['overlapping_subjects'],
        split_stats['overlapping_subjects']
    ]
    labels_overlap = ['Train Only', 'Val Only', 'Overlapping']
    colors_pie = [colors[0], colors[1], '#8B0000']  # Use darker red for overlapping
    wedges, texts, autotexts = ax4.pie(overlap_data, labels=labels_overlap, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax4.set_title('Subject Distribution in Train/Val Split', fontsize=14, fontweight='bold', pad=20)

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Plot 5: Sex distribution comparison (all three datasets)
    ax5 = axes[1, 1]
    sexes = ['F', 'M', 'Unknown']
    tuning_train_sex = [tuning_train_stats['sex_distribution'].get(sex, 0) for sex in sexes]
    tuning_val_sex = [tuning_val_stats['sex_distribution'].get(sex, 0) for sex in sexes]
    training_sex = [training_stats['sex_distribution'].get(sex, 0) for sex in sexes]

    x = np.arange(len(sexes))
    width = 0.25
    ax5.bar(x - width, tuning_train_sex, width, label='Tuning Train', color=colors[0], edgecolor='white', linewidth=1.2)
    ax5.bar(x, tuning_val_sex, width, label='Tuning Val', color=colors[1], edgecolor='white', linewidth=1.2)
    ax5.bar(x + width, training_sex, width, label='External Training', color=colors[2], edgecolor='white', linewidth=1.2)
    ax5.set_title('Sex Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(sexes, fontsize=10)
    ax5.tick_params(axis='y', labelsize=10)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Plot 6: Unique subjects percentage comparison
    ax6 = axes[1, 2]
    datasets_subjects = ['Tuning\nTrain', 'Tuning\nVal', 'External\nTraining']

    # Calculate percentages: unique subjects / total records * 100
    subject_percentages = [
        (tuning_train_stats['unique_subjects'] / tuning_train_stats['total_records']) * 100,
        (tuning_val_stats['unique_subjects'] / tuning_val_stats['total_records']) * 100,
        (training_stats['unique_subjects'] / training_stats['total_records']) * 100
    ]

    bars = ax6.bar(datasets_subjects, subject_percentages, color=colors, edgecolor='white', linewidth=1.2)
    ax6.set_title('Unique Subjects as % of Total Records', fontsize=14, fontweight='bold', pad=20)
    ax6.set_ylabel('Percentage (%)', fontsize=12)
    ax6.tick_params(axis='x', rotation=45, labelsize=10)
    ax6.tick_params(axis='y', labelsize=10)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, pct in zip(bars, subject_percentages):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(subject_percentages)*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Set overall figure background
    fig.patch.set_facecolor('white')

    # Save to output directory
    output_path = os.path.join(output_dir, 'cohort_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved as '{output_path}'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Cohort Analysis for MIMIC-IV Readmission Data')
    parser.add_argument('--tuning_data', default='/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json',
                        help='Path to tuning data file')
    parser.add_argument('--training_data', default='/ssd-shared/ryans_output/mimic_iv_readmission_train.json',
                        help='Path to training data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size for train/val split')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducible splits')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip creating visualizations')

    args = parser.parse_args()

    print("MIMIC-IV Readmission Data Cohort Analysis")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    tuning_data = load_json_data(args.tuning_data)
    training_data = load_json_data(args.training_data)

    # Analyze train/val split on tuning data first to get split datasets
    split_results = analyze_train_val_split(tuning_data, args.test_size, args.random_state)

    # Analyze cohorts for all three datasets (with 5k service samples)
    tuning_train_stats = analyze_cohort_characteristics(split_results['train_data'], "Tuning Train Dataset (80%)", service_sample_size=5000)
    tuning_val_stats = analyze_cohort_characteristics(split_results['val_data'], "Tuning Validation Dataset (20%)", service_sample_size=5000)
    training_stats = analyze_cohort_characteristics(training_data, "External Training Dataset", service_sample_size=5000)

    # Compare all three cohorts
    compare_cohorts(tuning_train_stats, tuning_val_stats, training_stats)

    # Create visualizations
    if not args.no_viz:
        try:
            create_visualizations(tuning_train_stats, tuning_val_stats, training_stats, split_results)
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    # Save summary report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'tuning_data_path': args.tuning_data,
        'training_data_path': args.training_data,
        'split_config': {'test_size': args.test_size, 'random_state': args.random_state},
        'tuning_train_stats': tuning_train_stats,
        'tuning_val_stats': tuning_val_stats,
        'training_stats': training_stats,
        'split_stats': split_results
    }

    with open('cohort_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis complete! Report saved as 'cohort_analysis_report.json'")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()