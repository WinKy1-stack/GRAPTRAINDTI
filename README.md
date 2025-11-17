<div align="center">

# 💊 GraphTransDTI: Drug-Target Interaction Prediction

### A State-of-the-Art Deep Learning Framework for *In Silico* Drug Discovery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

<p align="center">
  <em>"Unlocking the molecular language of life through Geometric Deep Learning and Attention Mechanisms."</em>
</p>

</div>

---

## 📋 Executive Summary

**GraphTransDTI** is an advanced end-to-end learning framework designed to predict the binding affinity between drug molecules and protein targets. By treating molecules as graphs and proteins as sequences, we leverage the power of **Graph Transformers** and **Bi-directional LSTMs** to capture intricate structural and sequential patterns.

The core innovation lies in our **Cross-Attention Interface**, which mimics the biological docking process computationally, allowing the model to focus on specific binding sites.

### 🚀 Key Objectives
- **Accuracy:** Surpass state-of-the-art (SOTA) baselines (DeepDTA, GraphDTA) by at least **10%** in RMSE.
- **Robustness:** Generalize well across diverse datasets (DAVIS, KIBA).
- **Efficiency:** Optimize inference time for high-throughput screening.

---

## 🧮 Methodology & Mathematical Formulation

### 1. Drug Encoding (Graph Transformer)
We represent drugs as molecular graphs $G = (V, E)$. To capture global structural information, we utilize Multi-Head Self-Attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $Q, K, V$ are feature matrices derived from atom nodes.

### 2. Protein Encoding (Sequence Modeling)
Target proteins are processed as amino acid sequences using a hybrid CNN-BiLSTM architecture to capture local motifs and long-range dependencies:

$$
h_t = \text{BiLSTM}(\text{CNN}(x_{protein}))
$$

### 3. Interaction Learning (Cross-Attention)
To model the binding affinity, we compute the interaction matrix between Drug features ($H_d$) and Protein features ($H_p$):

$$
\text{Interaction} = \text{MultiHead}(Q=H_d, K=H_p, V=H_p)
$$

Finally, the binding affinity score $\hat{y}$ is predicted via a Multi-Layer Perceptron (MLP).

---

## 🏗️ System Architecture

The detailed workflow of GraphTransDTI is illustrated below:

```mermaid
graph LR
    subgraph Input
    D[SMILES Drug]
    P[Protein Sequence]
    end

    subgraph Encoders
    D -->|RDKit| GE(Graph Transformer)
    P -->|Tokenize| PE(CNN + BiLSTM)
    end

    subgraph Fusion
    GE --> CA{Cross-Attention\nFusion Layer}
    PE --> CA
    end

    subgraph Output
    CA --> FCN[Fully Connected Network]
    FCN --> RES((Binding Affinity\npKd / KIBA))
    end
    
    style CA fill:#f96,stroke:#333,stroke-width:2px
    style RES fill:#9f9,stroke:#333,stroke-width:2px
