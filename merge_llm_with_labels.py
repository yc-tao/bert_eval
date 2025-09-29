#!/usr/bin/env python3
"""
Script to merge LLM-generated summaries with ground truth labels.

This script:
1. Loads ground truth data from mimic_iv_readmission_tuning.json
2. Parses LLM output from clinical_summaries.jsonl
3. Matches records by (subject_id, note_id)
4. Creates new dataset with LLM summaries and original metadata
5. Saves to /ssd-shared/yichen_data/
"""

import json
import sys
from pathlib import Path

def load_ground_truth(file_path):
    """Load ground truth JSON file."""
    print(f"Loading ground truth data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} ground truth records")
    return data

def create_subject_mapping(ground_truth_data):
    """Create mapping from (subject_id, note_id) to ground truth record."""
    print("Creating (subject_id, note_id) mapping...")
    mapping = {}
    for record in ground_truth_data:
        key = (record['subject_id'], record['note_id'])
        mapping[key] = record
    print(f"Created mapping for {len(mapping)} unique (subject_id, note_id) pairs")
    return mapping

def process_llm_data(llm_file_path, subject_mapping):
    """Process LLM JSONL file and create merged records."""
    print(f"Processing LLM data from {llm_file_path}...")

    merged_records = []
    processed_count = 0
    matched_count = 0

    with open(llm_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                try:
                    llm_record = json.loads(line.strip())
                    processed_count += 1

                    # Extract subject_id and note_id from the original field
                    subject_id = llm_record['original']['subject_id']
                    note_id = llm_record['original']['note_id']
                    key = (subject_id, note_id)

                    # Check if this record exists in ground truth
                    if key in subject_mapping:
                        matched_count += 1

                        # Get the ground truth record
                        gt_record = subject_mapping[key].copy()

                        # Replace the text with LLM-generated summary
                        gt_record['text'] = llm_record['text']

                        merged_records.append(gt_record)

                    # Progress indicator
                    if processed_count % 5000 == 0:
                        print(f"Processed {processed_count} records, matched {matched_count}")

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue

    print(f"Processing complete:")
    print(f"  Total LLM records processed: {processed_count}")
    print(f"  Records matched with ground truth: {matched_count}")
    print(f"  Records in final dataset: {len(merged_records)}")

    return merged_records

def save_merged_data(merged_records, output_path):
    """Save merged dataset to JSON file."""
    print(f"Saving merged dataset to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(merged_records, f, indent=2)

    print(f"Successfully saved {len(merged_records)} records to {output_path}")

def main():
    # File paths
    ground_truth_path = Path("/ssd-shared/yichen_data/mimic_iv_readmission_tuning.json")
    llm_output_path = Path("/home/grads/y/yichentao/locallighteval/outputs/clinical_summarization/2025-09-25_20-25-42/mimic_iv_readmission_tuning_clinical_summaries.jsonl")
    output_path = Path("/ssd-shared/yichen_data/mimic_iv_readmission_tuning_qwen_v2.json")

    # Validate input files exist
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        sys.exit(1)

    if not llm_output_path.exists():
        print(f"Error: LLM output file not found: {llm_output_path}")
        sys.exit(1)

    try:
        # Step 1: Load ground truth data
        ground_truth_data = load_ground_truth(ground_truth_path)

        # Step 2: Create subject_id mapping
        subject_mapping = create_subject_mapping(ground_truth_data)

        # Step 3: Process LLM data and create merged records
        merged_records = process_llm_data(llm_output_path, subject_mapping)

        # Step 4: Save merged dataset
        save_merged_data(merged_records, output_path)

        print("\n✅ Data merging completed successfully!")
        print(f"Output file: {output_path}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()