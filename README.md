<div align="center">

# 💊 GraphTransDTI: Drug-Target Interaction Prediction

### A Graph-based Deep Learning Framework for Drug Discovery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

**GraphTransDTI** is an advanced deep learning model designed to predict **Drug-Target Interactions (DTI)** with high precision. It leverages a hybrid architecture combining:

- **Graph Transformer** for drug molecules (SMILES → Molecular Graph).
- **CNN + BiLSTM** for protein targets (Amino acid sequences).
- **Cross-Attention Mechanism** to model the interaction interface.
- **Regression Head** to predict binding affinity metrics (KIBA, Kd, pKd).

### 🎯 Objectives
- **RMSE:** Reduce by ≥10% compared to baselines.
- **Pearson r:** Increase by ≥0.05.
- **Concordance Index (CI):** > 0.90.

### 🏆 Comparison with Baselines

| Feature | GraphTransDTI (Ours) | Traditional Baselines (DeepDTA) |
| :--- | :--- | :--- |
| **Drug Encoder** | **Graph Transformer** (Captures global structure) | CNN/GCN (Local features only) |
| **Protein Encoder** | **CNN + BiLSTM** (Seq & Context) | Simple CNN |
| **Fusion Strategy** | **Cross-Attention** (Interaction aware) | Concatenation (Naive) |
| **Complexity** | O(n²) attention | O(n) |

---

## 🔬 Model Architecture

The system takes a SMILES string and a Protein sequence as input, processes them through separate encoders, and fuses features using attention before prediction.

```mermaid
graph LR
    A[SMILES Drug] -->|RDKit| B(Graph Encoder\nGraph Transformer)
    C[Protein Sequence] -->|Tokenize| D(Protein Encoder\nCNN + BiLSTM)
    B --> E{Cross-Attention\nFusion Layer}
    D --> E
    E --> F[MLP Predictor]
    F --> G((Binding Affinity))
    style E fill:#f9f,stroke:#333,stroke-width:2px
