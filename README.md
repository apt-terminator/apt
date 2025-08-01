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

##LLM embeddings

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
