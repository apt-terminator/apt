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

