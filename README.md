# Cross-OS APT Detection via Multi-Path Anomaly Scoring

This repository provides the implementation and reproducibility guide for our work:  
**"Cross-OS Anomaly Detection via LLM-Powered Semantic Alignment and Geometric Scoring"**

## 🛡 Overview

We propose a novel unsupervised framework for detecting Advanced Persistent Threats (APTs) across different operating systems, leveraging:

- 🔹 **Path 1 — Semantic Similarity Scoring:** Using LLM embeddings (e.g., BERT, MiniLM, DistilBERT) of behavioral sentences describing processes.
- 🔹 **Path 2 — Structural Modeling:** Graph-based anomaly scoring via VGAE using cosine similarity graphs.
- 🔹 **Path 3 — Geometric Scoring with Optimal Transport (OT):** Including variants such as classic, entropy-regularized, angular-based, and local density-based OT.

Our final score is obtained via **max-fusion** across the three paths.
![Cross-OS Multi-Path Detection Pipeline](https://raw.githubusercontent.com/apt-terminator/apt/main/figures/figure1.png)


---

## 📁 Repository Structure

```bash
├── src/                        # Core implementation
│   ├── embeddings/             # LLM sentence encoder functions
│   ├── graphs/                 # VGAE graph construction and training
│   ├── ot/                     # Optimal Transport scoring variants
│   └── utils/                  # Data loaders, sentence construction, evaluation metrics
├── experiments/                # Reproducibility scripts for Attack Scenarios 1 & 2
├── figures/                    # Key visualizations (AUC, similarity heatmaps, etc.)
├── data/                       # Small toy example for demo (real data fetched separately)
├── README.md                   # You are here
└── requirements.txt            # Required Python packages


## 📊 Method Overview

![Cross-OS Multi-Path Detection Pipeline](https://raw.githubusercontent.com/apt-terminator/apt/refs/heads/main/figures/figure1.png)
![Cross-OS Multi-Path Detection Pipeline](https://raw.githubusercontent.com/apt-terminator/apt/main/figures/figure1.png)
*Figure 1: Overview of our cross-OS multi-path anomaly detection pipeline combining semantic similarity, structural modeling, and optimal transport scoring.*

![Cross-OS Multi-Path Detection Pipeline](https://github.com/apt-terminator/apt/blob/main/figures/figure1.png)

![Alt text](https://github.com/flagus-apt/flagus/blob/main/figures/AL_Windows_Pandex.png "Active Learning Windows Pandex")


