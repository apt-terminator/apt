#!/bin/bash

# Usage:
# ./bash.sh /Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT/data/bovia/5dir /Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT/data/bovia/5dir/5dir_bovia_simple.csv /Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT/data/bovia/cadets /Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT/data/bovia/cadets/cadets_bovia_webshell.csv /Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-TERMINATOR/Windows_to_BSD_exec_translation_dict.json

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

