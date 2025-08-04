# Cross-OS APT Detection via Multi-Path Anomaly Scoring

This repository provides the implementation and reproducibility guide for our work:  
**"Cross-OS Anomaly Detection via LLM-Powered Semantic Alignment and Geometric Scoring"**

## 🛡 Overview

We propose a novel unsupervised framework for detecting Advanced Persistent Threats (APTs) across different operating systems, leveraging:

- 🔹 **Path 1 — Semantic Similarity Scoring:** Using LLM embeddings (e.g., BERT, MiniLM, DistilBERT) of behavioral sentences describing processes.
- 🔹 **Path 2 — Structural Modeling:** Graph-based anomaly scoring via VGAE using cosine similarity graphs.
- 🔹 **Path 3 — Geometric Scoring with Optimal Transport (OT):** Including variants such as classic, entropy-regularized, angular-based, and local density-based OT.

Our final score is obtained via **max-fusion** across the three paths.

![Cross-OS Multi-Path Detection Pipeline](https://raw.githubusercontent.com/apt-terminator/apt/refs/heads/main/figures/figure1.png)



![Cross-OS Multi-Path Detection Pipeline](https://raw.githubusercontent.com/apt-terminator/apt/refs/heads/main/figures/figure2.png)

## DARPA Transparent Computing Data 

Databases are publically available at:  ![https://gitlab.com/adaptdata](https://gitlab.com/adaptdata)



---

## 📁 Repository Structure

```bash
├── src/                        # Core implementation
├── figures/                    # Key visualizations (AUC, similarity heatmaps, etc.)
├── data/                       # Small toy example for demo (real data fetched separately)
├── README.md                   # You are here
└── requirements.txt            # Required Python packages


```


## Requirements

To run this project, make sure you have the following dependencies installed:

- Python 3.8+
- PyTorch >= 1.12
- NumPy
- pandas
- scikit-learn
- torch-geometric
- matplotlib
- transformers
- networkx

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Pipeline via Bash
To simplify reproducibility, we provide a Bash wrapper script (`run_pipeline.sh`) that automates the execution of the full cross-OS anomaly detection pipeline.

### 🔧 Usage

```bash
bash run_pipeline.sh <source_dir> <source_ground_truth.csv> <target_dir> <target_ground_truth.csv> <translation_dict.json>
```
This script passes the required arguments to (`APT_Terminator_Similarity.py`) and echoes all parameters for verification. It ensures minimal setup and helps users easily reproduce the results reported in the paper. Each argument must be a valid path to the corresponding resource (directories, CSV files, or JSON dictionary).


## 🚀 Bash Script Usage Guide

To run the **APT Terminator pipeline** via bash with dictionary-based translation and OT scoring (MiniLM), use the following command structure:

### 📜 Bash Script

```bash
#!/bin/bash

# Usage:
# ./bash.sh /path/to/source_dir /path/to/source_gt.csv /path/to/target_dir /path/to/target_gt.csv /path/to/translation_dict.json

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

# Launch pipeline
python3.8 APT_Terminator_Dictionnary_OT_minilm.py \
  --source_directory "$source_directory" \
  --source_ground_truth "$source_ground_truth" \
  --target_directory "$target_directory" \
  --target_ground_truth "$target_ground_truth" \
  --dictionary "$dictionary"
```

## 🔁 Example: Cross-OS Transfer (Windows → BSD) – Attack Scenario 2

This example demonstrates how to run the APT Terminator pipeline transferring from **Windows** (source OS) to **BSD** (target OS) using **Attack Scenario 2** with the similarity based transfers.

### 📁 Dataset Paths

- **Source OS**: Windows  
  - 📂 Local (OpenReview): `./data/Windows`  
  - 🌐 GitHub: [Windows folder](https://github.com/apt-terminator/apt/tree/main/data/scenario2/windows)  
  - 📄 Ground Truth CSV:  
    - Local: `./data/Windows/5dir_bovia_simple.csv`  
    - GitHub: [5dir_bovia_simple.csv](https://github.com/apt-terminator/apt/blob/main/data/scenario2/windows/5dir_bovia_simple.csv)

- **Target OS**: BSD  
  - 📂 Local (OpenReview): `./data/BSD`  
  - 🌐 GitHub: [BSD folder](https://github.com/apt-terminator/apt/tree/main/data/scenario2/BSD)  
  - 📄 Ground Truth CSV:  
    - Local: `./data/BSD/cadets_bovia_webshell.csv`  
    - GitHub: [cadets_bovia_webshell.csv](https://github.com/apt-terminator/apt/blob/main/data/scenario2/BSD/cadets_bovia_webshell.csv)

- **Translation Dictionary (Windows → BSD)**  
  - 📂 Local: `./Windows_to_BSD_exec_translation_dict.json`  
  - 🌐 GitHub: [Translation Dictionary](https://github.com/apt-terminator/apt/blob/main/src/Windows_to_BSD_exec_translation_dict.json)

---

### ▶️ Run the Pipeline

Use the following Bash command to launch the pipeline:

```bash
bash run_pipeline.sh \
  ./data/Windows \
  ./data/Windows/5dir_bovia_simple.csv \
  ./data/BSD \
  ./data/BSD/cadets_bovia_webshell.csv \
  ./Windows_to_BSD_exec_translation_dict.json
```


## LLM embeddings
The following code example demonstrates how to generate contextualized sentence embeddings using a pre-trained Transformer model from the Hugging Face transformers library. It begins by selecting a model architecture (e.g., BERT, DistilBERT, or MiniLM), then loads the corresponding tokenizer and model weights. Given a sample sentence describing system behavior, the code tokenizes the input, feeds it through the model, and extracts the embedding associated with the [CLS] token — commonly used as a summary representation of the entire sentence in models like BERT. This embedding can be further used in downstream tasks such as semantic similarity, anomaly scoring, or classification.

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Choose your model
model_name = 'bert-base-uncased'  # options: 'distilbert-base-uncased', 'microsoft/MiniLM-L6-v2', etc.

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example sentence
sentence = "The process opened a file, connected to a remote host, and executed a binary."

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Use [CLS] token as sentence embedding (BERT-style)
embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_size]
```

## 🛡️ MITRE ATT&CK Mapping

To enhance interpretability and facilitate analyst trust, our framework integrates post hoc explainability via Large Language Models (LLMs), such as BERT and MiniLM. Each process is transformed into a behaviorally descriptive sentence and analyzed using LLM prompting to infer suspiciousness and align activities with known adversarial tactics.

The generated explanations are automatically mapped to the [MITRE ATT&CK](https://attack.mitre.org/) framework, helping bridge low-level trace signals with high-level adversary behavior.

| OS       | Behavior Summary                                                              | Mapped Tactics & Techniques                              | ATT&CK IDs                   |
|----------|-------------------------------------------------------------------------------|----------------------------------------------------------|------------------------------|
| **Linux**   | Opened file, executed binary, forked subprocess, renamed system file         | Masquerading, Process Injection, Shell Config Hijacking   | T1036.003, T1055, T1543.003  |
| **Windows** | Modified registry, created process, connected to host, wrote to file         | Registry Run Keys, Command Interpreter, Tool Transfer     | T1547.001, T1059, T1105      |
| **BSD**     | Changed file ownership and permissions before executing                     | Permission Modification, Sudo Abuse                       | T1222.002, T1548.001         |
| **Android** | Installed APK, sent data, held wakelock                                     | Malicious App Delivery, Sensitive Data Access, Autostart  | T1476, T1409, T1547          |

> 📌 These mappings are produced automatically from binary event vectors via sentence generation and LLM-based interpretation.

