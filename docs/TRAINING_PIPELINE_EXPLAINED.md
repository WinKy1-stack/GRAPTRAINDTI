# 📚 GRAPHTRANSDTI TRAINING PIPELINE - GIẢI THÍCH CHI TIẾT

## ✅ GPU SETUP HOÀN TẤT
```
GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
CUDA: 12.8 (Driver 572.83)
PyTorch: 2.5.1+cu121
Training Speed: Ước tính 2-3 giờ cho 100 epochs (nhanh hơn CPU 10-15x)
```

---

## 📁 CÁC FILE CẦN THIẾT ĐỂ TRAIN

### 1️⃣ **DATA FILES** (Đã có đủ)
```
data/kiba/
├── ligands_can.txt      # 2,111 SMILES (JSON format)
├── proteins.txt         # 229 protein sequences (JSON format)
└── Y                    # Affinity matrix (2111×229)

data/davis/
├── ligands_can.txt      # 68 SMILES
├── proteins.txt         # 442 sequences
└── Y                    # Affinity matrix (68×442)
```

**Vai trò**: Dữ liệu đầu vào - 118K drug-protein pairs cho training

---

### 2️⃣ **CONFIG FILE**
```
config.yaml
```

**Vai trò**: Hyperparameters - điều khiển toàn bộ quá trình training

**Các tham số quan trọng**:
```yaml
experiment:
  name: "GraphTransDTI_KIBA"
  device: "cuda"                    # ✅ Sử dụng GPU
  checkpoint_dir: "./checkpoints"   # Lưu model tốt nhất

training:
  batch_size: 64                    # 64 pairs/batch
  num_epochs: 100                   # Tối đa 100 epochs
  learning_rate: 0.0001
  early_stopping_patience: 15       # ✅ Stop nếu không improve 15 epochs

model:
  drug_encoder:
    hidden_dim: 128
    num_layers: 4                   # 4 Graph Transformer layers
    num_heads: 8                    # Multi-head attention
  
  protein_encoder:
    cnn_filters: [32, 64, 96]       # CNN cho protein motifs
    lstm_num_layers: 2              # BiLSTM cho context
  
  cross_attention:
    num_heads: 8                    # Drug-protein interaction
```

---

### 3️⃣ **DATALOADER FILES**

#### A. `src/dataloader/featurizer.py`
**Logic**: Chuyển đổi raw data → model-ready tensors

```python
SMILES string → RDKit Molecule → Graph
  ├── Node features: 78-dim (atom type, degree, aromatic, ...)
  ├── Edge features: 12-dim (bond type, conjugated, ring, ...)
  └── PyG Data object

Protein sequence → Token indices
  ├── 26 tokens: 20 amino acids + 5 special (PAD, UNK, ...)
  ├── Padding/Truncate to 1000 length
  └── Tensor [1000]
```

**Input Example**:
```python
SMILES: "CCO"                    # Ethanol
Protein: "MKVLWAALL..."          # 500 amino acids
Label: 12.5                      # KIBA score
```

**Output**:
```python
{
  'drug': PyG Data(x=[3, 78], edge_index=[2, 4], edge_attr=[4, 12]),
  'protein': Tensor[1000],       # Padded to 1000
  'label': Tensor[1]
}
```

#### B. `src/dataloader/kiba_loader.py`
**Logic**: Load KIBA dataset và split train/val/test

```python
1. Load JSON files
   ligands_dict = json.load('ligands_can.txt')   # {"CHEMBL123": "CCO", ...}
   proteins_dict = json.load('proteins.txt')
   affinity_matrix = pickle.load('Y')             # Shape (2111, 229)

2. Create pairs
   for i in range(2111):                          # Drugs
       for j in range(229):                       # Proteins
           if not np.isnan(affinity_matrix[i,j]):
               pairs.append((smiles[i], proteins[j], affinity_matrix[i,j]))
   
   → 118,254 valid pairs

3. Split dataset
   Random shuffle with seed=42
   80% train:  94,603 pairs
   10% val:    11,825 pairs
   10% test:   11,826 pairs

4. Create DataLoader
   Batch size: 64
   Collate function: collate_dti_batch()
```

#### C. `src/dataloader/davis_loader.py`
**Logic**: Tương tự KIBA, dùng cho generalization testing

---

### 4️⃣ **MODEL FILES**

#### A. `src/models/graph_transformer.py`
**Logic**: Encode drug molecules

```python
Graph Transformer Layer:
  Input: Node features [num_atoms, 78], Edges [2, num_bonds]
  
  Step 1: Multi-Head Attention
    Q, K, V = Linear(x)
    Attention(Q, K, V) = softmax(QK^T/√d) · V
    → Atoms attend to each other (learn molecular structure)
  
  Step 2: Feedforward Network
    FFN(x) = ReLU(Linear(x)) → Linear
  
  Step 3: Residual + LayerNorm
    Output = LayerNorm(x + Attention(x))
            + LayerNorm(x + FFN(x))

GraphTransformerEncoder:
  Embedding: [78] → [128]
  4× Transformer Layers
  Global Pooling: Average all atoms → [128]
  
  Output: Drug embedding [batch_size, 128]
```

**Ví dụ**:
```
Aspirin (CCO molecule):
  9 atoms → [9, 78] features
  → 4 Transformer layers (atoms talk to each other)
  → Average pooling → [128] vector (drug representation)
```

#### B. `src/models/protein_encoder.py`
**Logic**: Encode protein sequences

```python
ProteinEncoder:
  Step 1: Embedding
    Amino acid indices [batch, 1000] → [batch, 1000, 128]
  
  Step 2: CNN - Extract Local Motifs
    Conv1D kernel=4:  Capture 4-residue patterns (e.g., HELIX)
    Conv1D kernel=8:  Capture 8-residue patterns (e.g., BETA-SHEET)
    Conv1D kernel=12: Capture 12-residue patterns (e.g., DOMAINS)
    → Concatenate → [batch, 1000, 192] (32+64+96 filters)
  
  Step 3: BiLSTM - Capture Long-Range Dependencies
    Forward LSTM:  Read sequence left→right
    Backward LSTM: Read sequence right→left
    → Concatenate → [batch, 1000, 256] (128×2)
  
  Step 4: Global Pooling
    Average over sequence → [batch, 128]
  
  Output: Protein embedding [batch_size, 128]
```

**Ví dụ**:
```
Protein "MVKL..." (500 residues):
  → Embedding [500, 128]
  → CNN finds motifs (alpha-helix, beta-sheet)
  → BiLSTM captures long-range interactions
  → Average pooling → [128] vector (protein representation)
```

#### C. `src/models/cross_attention.py`
**Logic**: Learn drug-protein interactions

```python
CrossAttention:
  Input: Drug [batch, 128], Protein [batch, 128]
  
  Step 1: Expand to sequence
    Drug → [batch, 1, 128]     (treat as 1 "token")
    Protein → [batch, 1, 128]
  
  Step 2: Cross-Attention (Drug attends to Protein)
    Q = Drug
    K, V = Protein
    Attention = softmax(Q·K^T/√d) · V
    → Drug learns "which protein parts matter for binding"
  
  Step 3: Cross-Attention (Protein attends to Drug)
    Q = Protein
    K, V = Drug
    Attention = softmax(Q·K^T/√d) · V
    → Protein learns "which drug atoms matter for binding"
  
  Step 4: Fusion
    Concatenate [Drug_attended, Protein_attended] → [batch, 256]
    → Linear → [batch, 128]
  
  Output: Fused representation [batch_size, 128]
```

**Ví dụ**:
```
Aspirin + Protein interaction:
  Drug [128] + Protein [128]
  → Cross-attention: "Aspirin's OH group binds to Protein's active site"
  → Fused [128] (drug-protein interaction pattern)
```

#### D. `src/models/graphtransdti.py`
**Logic**: Complete end-to-end model

```python
GraphTransDTI.forward():
  Input: Drug graph, Protein sequence [batch, 1000]
  
  # Step 1: Encode Drug
  drug_emb = GraphTransformerEncoder(drug_graph)  # [batch, 128]
  
  # Step 2: Encode Protein
  protein_emb = ProteinEncoder(protein_seq)       # [batch, 128]
  
  # Step 3: Cross-Attention Fusion
  fused = CrossAttention(drug_emb, protein_emb)   # [batch, 128]
  
  # Step 4: Predict Binding Affinity
  x = Linear(fused, 256) → ReLU → Dropout
  x = Linear(x, 128) → ReLU → Dropout
  x = Linear(x, 64) → ReLU → Dropout
  prediction = Linear(x, 1)                       # [batch, 1]
  
  Output: KIBA score prediction (0-17 range)
```

**Ví dụ hoàn chỉnh**:
```
Input:  Aspirin + Target Protein
Output: 12.5 (KIBA score - binding affinity)

Flow:
  Aspirin SMILES → Graph [9 atoms]
    → Graph Transformer → [128] drug embedding
  
  Protein sequence [500 residues]
    → CNN+BiLSTM → [128] protein embedding
  
  Cross-Attention:
    → Learn "OH group ↔ active site" interaction
    → Fused [128]
  
  MLP Predictor:
    → [128] → [256] → [128] → [64] → [1]
    → Output: 12.5 (predicted KIBA score)
```

---

### 5️⃣ **TRAINING FILE**

#### `src/train.py`
**Logic**: Main training loop với early stopping

```python
Trainer.__init__():
  1. Set seed(42) for reproducibility
  2. Initialize model → GPU
  3. Setup optimizer (Adam, lr=0.0001)
  4. Setup scheduler (ReduceLROnPlateau)
  5. Load KIBA train/val dataloaders

Trainer.train_epoch():
  for batch in train_loader:
    # Forward pass
    predictions = model(drug_graph, protein_seq)
    
    # Compute loss
    loss = MSELoss(predictions, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(parameters, max_norm=1.0)  # Prevent exploding gradients
    optimizer.step()

Trainer.validate():
  with torch.no_grad():
    for batch in val_loader:
      predictions = model(drug_graph, protein_seq)
      loss = MSELoss(predictions, labels)
  
  # Calculate metrics
  RMSE = sqrt(mean((y_true - y_pred)²))
  Pearson = correlation(y_true, y_pred)
  CI = concordance_index(y_true, y_pred)

Trainer.train():
  for epoch in 1...100:
    # Train
    train_loss = train_epoch()
    
    # Validate
    val_loss, metrics = validate()
    
    # ✅ EARLY STOPPING LOGIC
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch
      patience_counter = 0
      
      # 💾 Save best model
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_metrics': metrics
      }, 'checkpoints/GraphTransDTI_KIBA_best.pt')
      
      print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    else:
      patience_counter += 1
    
    # 🛑 Stop if no improvement for 15 epochs
    if patience_counter >= 15:
      print(f"Early stopping at epoch {epoch}")
      print(f"Best epoch: {best_epoch}")
      break
```

**Training Example**:
```
Epoch 1/100
Train Loss: 2.5432
Val Loss: 2.3456 | RMSE: 1.5321 | Pearson: 0.6234 | CI: 0.7123
✓ Saved best model (val_loss: 2.3456)

Epoch 2/100
Train Loss: 2.1234
Val Loss: 2.1234 | RMSE: 1.4567 | Pearson: 0.6543 | CI: 0.7345
✓ Saved best model (val_loss: 2.1234)

...

Epoch 35/100
Train Loss: 0.8765
Val Loss: 1.0543 | RMSE: 1.0267 | Pearson: 0.8321 | CI: 0.8654
No improvement for 15 epochs
Early stopping triggered at epoch 35
Best epoch: 20 (val_loss: 0.9876)
```

---

### 6️⃣ **UTILITY FILES**

#### A. `src/utils/metrics.py`
**Logic**: Evaluation metrics

```python
RMSE (Root Mean Squared Error):
  RMSE = sqrt(mean((y_true - y_pred)²))
  → Lower is better (0 = perfect)
  → Measures prediction accuracy

Pearson Correlation:
  r = cov(y_true, y_pred) / (std(y_true) × std(y_pred))
  → Range: -1 to 1 (1 = perfect linear correlation)
  → Measures how well predictions follow true values

Concordance Index (CI):
  CI = P(y_pred_i > y_pred_j | y_true_i > y_true_j)
  → Range: 0 to 1 (1 = perfect ranking)
  → Measures ranking quality (important for drug screening)
```

#### B. `src/utils/seed.py`
**Logic**: Reproducibility

```python
set_seed(42):
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  
  → Same results every run (important for thesis)
```

#### C. `src/utils/smiles_to_graph.py`
**Logic**: SMILES → PyG Graph conversion

```python
smiles_to_graph("CCO"):
  # Step 1: Parse SMILES
  mol = Chem.MolFromSmiles("CCO")
  
  # Step 2: Extract atom features
  for atom in mol.GetAtoms():
    features = [
      atom.GetAtomicNum(),           # Element (6=C, 8=O)
      atom.GetDegree(),              # Number of bonds
      atom.GetTotalValence(),        # Valence
      atom.GetIsAromatic(),          # Aromatic?
      atom.IsInRing(),               # In ring?
      ...
    ]  # 78 features total
  
  # Step 3: Extract bond features
  for bond in mol.GetBonds():
    edge_features = [
      bond.GetBondType(),            # Single/Double/Triple
      bond.GetIsConjugated(),        # Conjugated?
      bond.IsInRing(),               # In ring?
      ...
    ]  # 12 features total
  
  # Step 4: Create PyG Data
  data = Data(
    x = atom_features,               # [num_atoms, 78]
    edge_index = [[0,1], [1,2], ...],# [2, num_bonds]
    edge_attr = edge_features        # [num_bonds, 12]
  )
```

---

### 7️⃣ **EVALUATION FILE**

#### `src/evaluate.py`
**Logic**: Test trên DAVIS dataset (generalization)

```python
Evaluator.__init__():
  1. Load trained model from checkpoint
  2. Load DAVIS test dataloader

Evaluator.evaluate():
  with torch.no_grad():
    for batch in davis_test_loader:
      predictions = model(drug, protein)
      store predictions and labels
  
  # Calculate metrics on DAVIS
  metrics = calculate_metrics(all_labels, all_predictions)
  
  print("DAVIS Test Results:")
  print(f"RMSE: {metrics['rmse']:.4f}")
  print(f"Pearson: {metrics['pearson']:.4f}")
  print(f"CI: {metrics['ci']:.4f}")
```

**Usage**:
```bash
python src/evaluate.py \
  --checkpoint checkpoints/GraphTransDTI_KIBA_best.pt \
  --dataset davis \
  --split test
```

---

## 🔄 TOÀN BỘ TRAINING PIPELINE

### **Flowchart**:
```
1. Load Data
   ├── data/kiba/ligands_can.txt (2,111 SMILES)
   ├── data/kiba/proteins.txt (229 sequences)
   └── data/kiba/Y (118,254 pairs)

2. Featurization (for each pair)
   ├── SMILES → RDKit → Graph [atoms, bonds]
   │   ├── Node features: [num_atoms, 78]
   │   └── Edge features: [num_edges, 12]
   └── Protein → Token indices → [1000]

3. Model Forward Pass
   ├── Drug: Graph → Graph Transformer → [128]
   ├── Protein: Sequence → CNN+BiLSTM → [128]
   ├── Fusion: Cross-Attention → [128]
   └── Prediction: MLP → [1] (KIBA score)

4. Training Loop (for each epoch)
   ├── Train Phase:
   │   ├── Forward pass → predictions
   │   ├── Compute loss = MSE(predictions, labels)
   │   ├── Backward pass → gradients
   │   └── Update weights
   │
   └── Validation Phase:
       ├── Forward pass (no gradients)
       ├── Compute metrics (RMSE, Pearson, CI)
       └── Check early stopping:
           ├── If val_loss improved → Save model
           └── If no improvement for 15 epochs → STOP

5. Output
   ├── checkpoints/GraphTransDTI_KIBA_best.pt (best model)
   ├── training_history.pkl (loss curves)
   └── logs/ (tensorboard logs)
```

---

## 📊 KẾT QUẢ MONG ĐỢI

### Training (KIBA):
- **Best Epoch**: ~30-50 (with early stopping)
- **Training Time**: 2-3 giờ trên RTX 3050
- **Best Val Loss**: ~0.8-1.0
- **Val Metrics**:
  - RMSE: 0.9-1.1 (↓ càng thấp càng tốt)
  - Pearson: 0.82-0.88 (↑ càng cao càng tốt)
  - CI: 0.85-0.90 (↑ càng cao càng tốt)

### Generalization (DAVIS):
- **Test Metrics**:
  - RMSE: 1.0-1.2
  - Pearson: 0.78-0.84
  - CI: 0.82-0.87

### So sánh với Baselines:
```
Model            | RMSE (KIBA) | Pearson | CI
-----------------+-------------+---------+------
DeepDTA          | 1.15        | 0.78    | 0.82
GraphDTA         | 1.05        | 0.83    | 0.85
GraphTransDTI    | 0.95        | 0.87    | 0.88  ← MỤC TIÊU
(10% improvement)
```

---

## 🚀 CÁCH CHẠY TRAINING

### Quick Test (5 epochs):
```bash
# Edit config.yaml: num_epochs: 5
python src/train.py
```

### Full Training (100 epochs với early stopping):
```bash
python src/train.py
# Sẽ chạy 2-3 giờ
# Tự động stop nếu không improve sau 15 epochs
```

### Evaluate on DAVIS:
```bash
python src/evaluate.py \
  --checkpoint checkpoints/GraphTransDTI_KIBA_best.pt \
  --dataset davis \
  --split test
```

---

## 💾 OUTPUT FILES

Sau khi training xong:
```
checkpoints/
└── GraphTransDTI_KIBA_best.pt         # Model tốt nhất (dùng cho thesis)
    ├── model_state_dict               # Model weights
    ├── epoch                          # Epoch nào đạt best
    ├── val_loss                       # Val loss tốt nhất
    └── val_metrics                    # RMSE, Pearson, CI

logs/
└── training.log                       # Chi tiết từng epoch

training_history.pkl                   # Loss curves (dùng để vẽ đồ thị)
```

---

## 📈 VẼ ĐỒ THỊ

Sử dụng `src/plot_results.py`:
```python
import pickle
import matplotlib.pyplot as plt

# Load history
with open('checkpoints/GraphTransDTI_KIBA_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot loss curves
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
```

---

## ✅ CHECKLIST TRƯỚC KHI TRAIN

- [x] GPU setup (CUDA available)
- [x] PyTorch 2.5.1+cu121 installed
- [x] Data downloaded (KIBA: 118K pairs)
- [x] Dataloaders verified
- [x] Model architecture complete
- [x] Config.yaml ready
- [x] Early stopping implemented
- [ ] Ready to start training!

---

## 🎯 TÓM TẮT

**CÁC FILE CHÍNH**:
1. `config.yaml` - Hyperparameters
2. `src/train.py` - Training script ✅ Early stopping
3. `src/dataloader/` - Load & featurize data
4. `src/models/` - GraphTransDTI architecture
5. `src/utils/` - Metrics, seed, SMILES converter
6. `src/evaluate.py` - Test on DAVIS

**LOGIC**:
- SMILES → Graph (78-dim atoms, 12-dim bonds)
- Protein → Tokens (26 amino acids, pad to 1000)
- Graph Transformer (4 layers) → Drug [128]
- CNN+BiLSTM → Protein [128]
- Cross-Attention → Fusion [128]
- MLP → Binding affinity [1]
- Early stopping: Lưu model tốt nhất, dừng nếu 15 epochs không improve

**EXPECTED TIME**: 2-3 giờ trên RTX 3050

**READY TO TRAIN!** 🚀
