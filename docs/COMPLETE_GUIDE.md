# HƯỚNG DẪN HOÀN CHỈNH - GraphTransDTI
## Ứng dụng mô hình dựa trên đồ thị cho khám phá và dự đoán thuốc trong y dược

---

## 📚 MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Cài đặt](#2-cài-đặt)
3. [Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu)
4. [Chạy training](#4-chạy-training)
5. [Evaluation](#5-evaluation)
6. [Kết quả mong đợi](#6-kết-quả-mong-đợi)
7. [Troubleshooting](#7-troubleshooting)
8. [Báo cáo đồ án](#8-báo-cáo-đồ-án)

---

## 1. Giới thiệu

### Mục tiêu đồ án
Xây dựng mô hình **GraphTransDTI** để dự đoán **tương tác thuốc-protein (DTI)** với độ chính xác cao hơn các baseline hiện có.

### Đóng góp chính
- ✅ **Graph Transformer** cho drug (thay vì GCN/GAT)
- ✅ **CNN + BiLSTM** cho protein (thay vì CNN/LSTM đơn)
- ✅ **Cross-Attention** fusion (thay vì concat)
- ✅ Đánh giá trên **KIBA** (train) & **DAVIS** (generalization)

### Kết quả mong đợi
- **RMSE**: Giảm ≥10% so với GraphDTA
- **Pearson r**: Tăng ≥0.05
- **CI**: > 0.90

---

## 2. Cài đặt

### Bước 1: Clone repository
```bash
cd C:\Workspace\DACNTT_Nhu
# (hoặc nơi bạn đã clone)
```

### Bước 2: Tạo môi trường ảo
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Bước 3: Cài đặt dependencies
```powershell
cd GraphTransDTI
pip install -r src/requirements.txt
```

**Nếu PyTorch Geometric lỗi**:
```powershell
# Install PyTorch với CUDA (nếu có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Bước 4: Kiểm tra cài đặt
```powershell
python check_installation.py
```

Kết quả mong đợi:
```
✓ PyTorch                  | Version: 2.0.0
✓ PyTorch Geometric        | Version: 2.3.0
✓ RDKit                    | Version: 2022.9.1
...
✓ All dependencies are installed correctly!
```

---

## 3. Chuẩn bị dữ liệu

### KIBA Dataset (bắt buộc)

#### Option 1: Download từ GitHub
```powershell
cd data\kiba

# Download ligands
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/ligands_can.txt" -OutFile "ligands_can.txt"

# Download proteins
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/proteins.txt" -OutFile "proteins.txt"

# Download affinity matrix
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/Y" -OutFile "Y"

cd ..\..
```

#### Option 2: Download thủ công
1. Vào https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba
2. Download 3 files: `ligands_can.txt`, `proteins.txt`, `Y`
3. Đặt vào `GraphTransDTI/data/kiba/`

### DAVIS Dataset (để test generalization)
```powershell
cd data\davis

Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/ligands_can.txt" -OutFile "ligands_can.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/proteins.txt" -OutFile "proteins.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/Y" -OutFile "Y"

cd ..\..
```

### Kiểm tra dữ liệu
```powershell
cd src
python -c "from dataloader import get_kiba_dataloader; get_kiba_dataloader('../data/kiba', 'train', batch_size=4, num_workers=0)"
```

---

## 4. Chạy training

### Option 1: Command line (khuyến nghị)
```powershell
cd src
python train.py
```

**Output mong đợi**:
```
==========================================
GraphTransDTI Training
==========================================
Experiment: GraphTransDTI_KIBA
Dataset: KIBA
Device: cuda
==========================================

[INFO] Model parameters: 2,234,567
[INFO] KIBA TRAIN dataset loaded: 94603 pairs
[INFO] KIBA VAL dataset loaded: 11825 pairs

========================================
Epoch 1/100
========================================
Training: 100%|████████| 1478/1478 [03:21<00:00]
Train Loss: 0.4521
Validating: 100%|████████| 185/185 [00:32<00:00]
Val Loss: 0.3876

==========================================
Validation Metrics:
==========================================
RMSE:            0.6224
Pearson r:       0.8543 (p=0.0e+00)
Concordance Index: 0.8821
==========================================

✓ Saved best model (val_loss: 0.3876)
...
```

### Option 2: Jupyter Notebook
1. Mở `notebooks/Train_GraphTransDTI.ipynb`
2. Chạy từng cell theo thứ tự

### Tùy chỉnh hyperparameters
Sửa file `config.yaml`:
```yaml
training:
  batch_size: 64        # Giảm nếu hết RAM
  learning_rate: 0.0001
  num_epochs: 100       # Giảm xuống 10 để test nhanh
```

---

## 5. Evaluation

### Test trên KIBA (test set)
```powershell
python evaluate.py --checkpoint ..\checkpoints\GraphTransDTI_KIBA_best.pt --dataset kiba --split test
```

### Test trên DAVIS (generalization)
```powershell
python evaluate.py --checkpoint ..\checkpoints\GraphTransDTI_KIBA_best.pt --dataset davis --split test
```

### Visualize results
```powershell
python plot_results.py
```

---

## 6. Kết quả mong đợi

### KIBA (Training dataset)

| Metric | Target | Thực tế (sau training) |
|--------|--------|------------------------|
| RMSE | < 0.370 | _[Ghi kết quả của bạn]_ |
| Pearson r | > 0.88 | _[Ghi kết quả của bạn]_ |
| CI | > 0.90 | _[Ghi kết quả của bạn]_ |

### DAVIS (Generalization)

| Metric | Target | Thực tế |
|--------|--------|---------|
| RMSE | < 0.270 | _[Ghi]_ |
| Pearson r | > 0.89 | _[Ghi]_ |
| CI | > 0.89 | _[Ghi]_ |

### So sánh với baseline

| Model | RMSE (KIBA) | Pearson r | CI |
|-------|-------------|-----------|-----|
| DeepDTA | 0.420 | 0.863 | 0.878 |
| GraphDTA | 0.398 | 0.876 | 0.889 |
| MolTrans | 0.385 | 0.884 | 0.895 |
| **GraphTransDTI** | **_[Ghi]_** | **_[Ghi]_** | **_[Ghi]_** |

---

## 7. Troubleshooting

### Lỗi: "CUDA out of memory"
**Giải pháp**:
```yaml
# Giảm batch_size trong config.yaml
training:
  batch_size: 32  # hoặc 16
```

### Lỗi: "RDKit invalid SMILES"
**Giải pháp**: Một số SMILES không hợp lệ sẽ tự động bỏ qua. Nếu quá nhiều:
```python
# Check trong src/dataloader/featurizer.py
# Line ~90: return None if invalid
```

### Lỗi: "pickle.UnpicklingError"
**Giải pháp**:
```python
# Trong kiba_loader.py, thêm encoding
with open(affinity_file, 'rb') as f:
    affinity_matrix = pickle.load(f, encoding='latin1')
```

### Training quá lâu
**Giải pháp**:
1. Giảm `num_epochs` xuống 10 để test
2. Sử dụng GPU (nếu có)
3. Tăng `batch_size` (nếu đủ RAM)

### Muốn chạy nhanh để demo
```yaml
# config.yaml
training:
  batch_size: 128      # Tăng
  num_epochs: 10       # Giảm
data:
  train_ratio: 0.1     # Chỉ dùng 10% data
```

---

## 8. Báo cáo đồ án

### Cấu trúc báo cáo (Word/LaTeX)

#### Chương 1: Giới thiệu
- Bối cảnh: Drug discovery tốn kém
- Bài toán: Dự đoán DTI
- Đóng góp: GraphTransDTI

#### Chương 2: Cơ sở lý thuyết
- Graph Neural Networks
- Transformer & Attention
- DTI prediction

#### Chương 3: Các hướng ứng dụng GNN trong y dược
- Molecular property prediction
- Drug-Drug Interaction
- **Drug-Target Interaction** ← chọn
- Drug-Disease association

#### Chương 4: Tổng quan nghiên cứu liên quan
- DeepDTA (2018)
- GraphDTA (2020)
- MolTrans (2022)
- Graphormer-DTI (2023)
- **Khoảng trống**: Chưa có Cross-Attention

#### Chương 5: Phương pháp đề xuất
- Kiến trúc GraphTransDTI
- Graph Transformer
- CNN + BiLSTM
- Cross-Attention
- Dataset: KIBA, DAVIS

#### Chương 6: Thực nghiệm
- Setup: GPU, PyTorch
- Hyperparameters
- Training process
- **Kết quả**:
  - Bảng so sánh
  - Biểu đồ (training curve, scatter plot)
  - Phân tích

#### Chương 7: Kết luận & Hướng phát triển
- Tóm tắt đóng góp
- Hạn chế
- Future work: 3D structure, pre-training, interpretability

### Tài liệu tham khảo (≥15 papers)
- [1] Tang et al. (2014) KIBA
- [2] Davis et al. (2011) DAVIS
- [3] Öztürk et al. (2018) DeepDTA
- [4] Nguyen et al. (2021) GraphDTA
- [5] Huang et al. (2022) MolTrans
- [6] Vaswani et al. (2017) Attention
- [7] Ying et al. (2021) Transformers for Graphs
- ... (còn 8 papers nữa)

### Hình vẽ cần có
1. **Sơ đồ kiến trúc tổng thể** (draw.io)
2. **Graph Transformer layer** (chi tiết)
3. **Protein encoder** (CNN + BiLSTM)
4. **Cross-Attention mechanism**
5. **Training curve** (loss, RMSE, Pearson)
6. **Scatter plot** (predicted vs true)
7. **Comparison bar chart** (baseline)

### File báo cáo
- `docs/BaoCao_DoAn.docx` (hoặc .tex)
- `docs/Slide_BaoVe.pptx` (10-15 slide)

---

## 9. Checklist hoàn thành đồ án

### Code ✅
- [x] Model architecture (4 files)
- [x] Dataloader (KIBA, DAVIS)
- [x] Training script
- [x] Evaluation script
- [x] Visualization
- [x] Notebook demo

### Dữ liệu ✅
- [ ] Download KIBA
- [ ] Download DAVIS
- [ ] Test dataloader

### Experiments
- [ ] Train GraphTransDTI trên KIBA (100 epochs)
- [ ] Evaluate trên KIBA test
- [ ] Evaluate trên DAVIS test
- [ ] So sánh với baseline (tìm số liệu từ papers)
- [ ] Tạo biểu đồ, bảng

### Báo cáo
- [ ] Viết Chương 1-7
- [ ] Vẽ sơ đồ kiến trúc
- [ ] Thêm hình ảnh kết quả
- [ ] Trích dẫn tài liệu tham khảo
- [ ] Làm slide thuyết trình

### Kiểm tra cuối
- [ ] Code chạy được (test lại từ đầu)
- [ ] README.md đầy đủ
- [ ] Báo cáo không lỗi chính tả
- [ ] Slide dưới 15 phút

---

## 10. Liên hệ & Hỗ trợ

### Tài liệu
- **README.md**: Hướng dẫn tổng quan
- **docs/MODEL_ARCHITECTURE.md**: Chi tiết kiến trúc
- **data/DATA_DOWNLOAD_GUIDE.md**: Hướng dẫn tải data
- **notebooks/Train_GraphTransDTI.ipynb**: Demo notebook

### Code structure
```
GraphTransDTI/
├── src/
│   ├── models/          ← Kiến trúc mô hình
│   ├── dataloader/      ← Xử lý dữ liệu
│   ├── utils/           ← Metrics, visualization
│   ├── train.py         ← Training script
│   └── evaluate.py      ← Evaluation script
├── data/                ← Datasets
├── notebooks/           ← Jupyter notebooks
├── config.yaml          ← Hyperparameters
└── README.md
```

---

## 🎯 Kết luận

**GraphTransDTI** là một đồ án tốt nghiệp hoàn chỉnh về:
- ✅ Deep Learning (Graph Transformer, Attention)
- ✅ Bioinformatics (Drug-Target Interaction)
- ✅ Software Engineering (clean code, documentation)

**Chúc bạn bảo vệ thành công! 🎓🚀**

---

**Cập nhật**: 2025-01-14  
**Version**: 1.0
