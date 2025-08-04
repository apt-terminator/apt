#!/bin/bash

# Usage:
# ./run_pipeline.sh ./data/windows ./data/windows/5dir_bovia_simple.csv ./data/BSD ./data/BSD/cadets_bovia_webshell.csv ./data/Windows_to_BSD_exec_translation_dict.json

source_directory=$1
source_ground_truth=$2
target_directory=$3
target_ground_truth=$4
dictionary=$5


echo "Running pipeline with:"
echo "  source_directory: $source_directory"
echo "  source_ground_truth: $source_ground_truth"
echo "  target_directory: $target_directory"
echo "  target_ground_truth: $target_ground_truth"
echo "  dictionary: $dictionary"

# Run Python script
python3.8 APT_Terminator_Similarity.py \
  --source_directory "$source_directory" \
  --source_ground_truth "$source_ground_truth" \
  --target_directory "$target_directory" \
  --target_ground_truth "$target_ground_truth" \
  --dictionary "$dictionary"

