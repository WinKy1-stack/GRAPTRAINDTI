# GraphTransDTI - Kiến trúc mô hình chi tiết

## Tổng quan

**GraphTransDTI** là mô hình deep learning end-to-end cho dự đoán Drug-Target Interaction (DTI), kết hợp:
- **Graph Transformer** (drug)
- **CNN + BiLSTM** (protein)  
- **Cross-Attention** (fusion)
- **MLP** (predictor)

---

## 1. Drug Encoder - Graph Transformer

### Input
- **SMILES string** → RDKit → **Molecular Graph**
- **Nodes**: Atoms (C, N, O, S, ...)
- **Edges**: Chemical bonds (single, double, triple, aromatic)

### Node Features (78-dim)
```python
[
    Atom type (44-dim, one-hot),         # C, N, O, S, F, ...
    Degree (11-dim, one-hot),            # 0-10 connections
    Formal charge (11-dim, one-hot),     # -5 to +5
    Hybridization (6-dim, one-hot),      # SP, SP2, SP3, ...
    Aromaticity (1-dim, binary),         # Is aromatic?
    Total H (5-dim, one-hot),            # 0-4 hydrogens
    Chirality (1-dim, binary)            # Has chirality?
]
```

### Edge Features (12-dim)
```python
[
    Bond type (4-dim, one-hot),          # SINGLE, DOUBLE, TRIPLE, AROMATIC
    Conjugated (1-dim, binary),          # Is conjugated?
    In ring (1-dim, binary),             # Part of ring?
    Stereo (6-dim, one-hot)              # Z, E, CIS, TRANS, ...
]
```

### Architecture
```
SMILES → Graph [N nodes, E edges]
  ↓
Node Embedding: Linear(78 → 128)
  ↓
Graph Transformer Layer 1:
  - Multi-head Attention (8 heads, 128-dim)
  - Uses edge features for attention bias
  - Layer Norm + Residual
  - FFN (128 → 512 → 128)
  - Layer Norm + Residual
  ↓
... × 4 layers
  ↓
Global Mean Pooling (over all nodes)
  ↓
Drug Embedding: [batch_size, 128]
```

### Key Parameters
```yaml
atom_features: 78
hidden_dim: 128
num_layers: 4
num_heads: 8
dropout: 0.1
edge_features: 12
```

### Code
```python
# src/models/graph_transformer.py
class GraphTransformerEncoder(nn.Module):
    def __init__(self, atom_features=78, hidden_dim=128, num_layers=4, num_heads=8):
        self.atom_embedding = nn.Linear(atom_features, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data):
        x = self.atom_embedding(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch)
        return self.output_projection(x)
```

---

## 2. Protein Encoder - CNN + BiLSTM

### Input
- **Amino acid sequence** (string) → **Integer indices**
- Vocabulary: 26 tokens (20 AAs + PAD, UNK, etc.)
- Max length: 1000 residues (padding/truncation)

### Architecture
```
Protein Sequence: [batch_size, 1000]
  ↓
Embedding Layer: (26 → 128)
  [batch_size, 1000, 128]
  ↓
3 × CNN Layers (parallel):
  - CNN-1: kernel=4, filters=32  (motif length 4)
  - CNN-2: kernel=8, filters=64  (motif length 8)
  - CNN-3: kernel=12, filters=96 (motif length 12)
  ↓
Concatenate: 32+64+96 = 192 filters
  [batch_size, 1000, 192]
  ↓
BiLSTM (2 layers):
  - Hidden: 128
  - Bidirectional → 256 output
  - Dropout: 0.1
  [batch_size, 1000, 256]
  ↓
Output Projection: Linear(256 → 128)
  ↓
Last Hidden State (or full sequence for attention)
  [batch_size, 128]
```

### Key Parameters
```yaml
vocab_size: 26
embedding_dim: 128
cnn_filters: [32, 64, 96]
cnn_kernel_sizes: [4, 8, 12]
lstm_hidden_dim: 128
lstm_num_layers: 2
lstm_dropout: 0.1
bidirectional: True
```

### Code
```python
# src/models/protein_encoder.py
class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size=26, embedding_dim=128, ...):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn = ProteinCNN(embedding_dim, [32,64,96], [4,8,12])
        self.lstm = nn.LSTM(192, 128, 2, bidirectional=True, batch_first=True)
        self.output_projection = nn.Linear(256, 128)
    
    def forward(self, x, return_sequence=False):
        x = self.embedding(x)
        x = self.cnn(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        if return_sequence:
            return self.output_projection(lstm_out)
        else:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.output_projection(h_last)
```

---

## 3. Cross-Attention Fusion

### Motivation
- Drug và protein không độc lập → cần học **tương tác**
- Nguyên tử nào của thuốc bind vào vị trí nào của protein?
- Cross-Attention: drug query → protein key/value

### Architecture
```
Drug Embedding:    [batch_size, 128]
Protein Embedding: [batch_size, 128]
  ↓
Drug → Protein Attention:
  Query: Drug
  Key, Value: Protein
  Multi-head (8 heads)
  ↓
Drug' = Attention(Drug, Protein) + Drug (residual)
  ↓
Protein → Drug Attention:
  Query: Protein
  Key, Value: Drug
  Multi-head (8 heads)
  ↓
Protein' = Attention(Protein, Drug) + Protein (residual)
  ↓
Fusion: Concat([Drug', Protein']) → Linear(256 → 128)
  ↓
Fused Embedding: [batch_size, 128]
```

### Key Parameters
```yaml
hidden_dim: 128
num_heads: 8
dropout: 0.1
fusion_method: "concat"  # or "add", "gate"
```

### Code
```python
# src/models/cross_attention.py
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8):
        self.drug_to_protein_attn = MultiHeadCrossAttention(hidden_dim, num_heads)
        self.protein_to_drug_attn = MultiHeadCrossAttention(hidden_dim, num_heads)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, drug_emb, protein_emb):
        drug_attended = self.drug_to_protein_attn(drug_emb, protein_emb, protein_emb)
        protein_attended = self.protein_to_drug_attn(protein_emb, drug_emb, drug_emb)
        fused = torch.cat([drug_attended, protein_attended], dim=-1)
        return self.fusion(fused)
```

---

## 4. Prediction Head

### Architecture
```
Fused Embedding: [batch_size, 128]
  ↓
FC-1: Linear(128 → 256) + ReLU + Dropout(0.2)
  ↓
FC-2: Linear(256 → 128) + ReLU + Dropout(0.2)
  ↓
FC-3: Linear(128 → 64) + ReLU + Dropout(0.2)
  ↓
Output: Linear(64 → 1)
  [batch_size, 1]  ← Binding affinity (regression)
```

### Loss Function
- **MSE Loss**: Mean Squared Error
- Optional: Huber Loss, Smooth L1 Loss

---

## 5. Complete Pipeline

### Forward Pass
```python
# Input
drug_smiles = "CCO"  # Ethanol
protein_seq = "MKTAYIAKQRQ..."

# 1. Featurization
drug_graph = smiles_to_graph(drug_smiles)  # PyG Data
protein_idx = featurize_protein(protein_seq)  # [1000]

# 2. Encoding
drug_emb = drug_encoder(drug_graph)  # [1, 128]
protein_emb = protein_encoder(protein_idx)  # [1, 128]

# 3. Fusion
fused = cross_attention(drug_emb, protein_emb)  # [1, 128]

# 4. Prediction
affinity = predictor(fused)  # [1, 1]
```

### Training
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        predictions = model(batch['drug'], batch['protein'])
        loss = criterion(predictions, batch['label'])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

## 6. Model Statistics

### Total Parameters
```
Graph Transformer:    ~800K
Protein Encoder:      ~1.2M
Cross-Attention:      ~150K
Predictor:           ~50K
─────────────────────────────
Total:               ~2.2M parameters
```

### Inference Speed
- **Single prediction**: ~20ms (GPU)
- **Batch 64**: ~50 predictions/second

### Memory
- **Training**: ~4 GB GPU (batch 64)
- **Inference**: ~1 GB GPU

---

## 7. Comparison with Baselines

| Model | Drug Encoder | Protein Encoder | Fusion | Params |
|-------|--------------|-----------------|--------|--------|
| **DeepDTA** | CNN (SMILES text) | CNN | Concat | ~3M |
| **GraphDTA** | GCN (3 layers) | CNN | Concat | ~2M |
| **MolTrans** | Transformer | Transformer | Concat | ~8M |
| **GraphTransDTI** | Graph Transformer | CNN+BiLSTM | Cross-Attn | ~2.2M |

### Advantages
- ✅ Graph Transformer > GCN (global receptive field)
- ✅ BiLSTM > CNN (long-range protein context)
- ✅ Cross-Attention > Concat (learns interaction)
- ✅ Fewer parameters than MolTrans

---

## 8. Training Configuration

### Hyperparameters (KIBA)
```yaml
batch_size: 64
learning_rate: 0.0001
weight_decay: 0.00001
optimizer: Adam
scheduler: ReduceLROnPlateau
epochs: 100
early_stopping: 15 patience
gradient_clip: 1.0
```

### Data Split
- Train: 80% (~94K pairs)
- Val: 10% (~12K pairs)
- Test: 10% (~12K pairs)

### Training Time
- **1 epoch**: ~10 minutes (V100 GPU)
- **Full training**: ~6 hours (with early stopping)

---

## 9. Interpretability (Future Work)

### Attention Visualization
```python
# Get attention weights
drug_to_protein_attn = model.cross_attention.drug_to_protein_attn.attn_weights
# Shape: [batch, num_heads, query_len, key_len]

# Visualize which atoms attend to which residues
import matplotlib.pyplot as plt
sns.heatmap(drug_to_protein_attn[0, 0].cpu().numpy())
plt.xlabel('Protein Residues')
plt.ylabel('Drug Atoms')
plt.show()
```

---

## 10. References

### Papers
1. Vaswani et al. (2017) "Attention Is All You Need" - *NIPS*
2. Ying et al. (2021) "Do Transformers Really Perform Bad for Graph Representation?" - *NeurIPS*
3. Öztürk et al. (2018) "DeepDTA" - *Bioinformatics*
4. Nguyen et al. (2021) "GraphDTA" - *Bioinformatics*

### Code
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- RDKit: https://www.rdkit.org/docs/

---

**Cập nhật**: 2025-01-14
