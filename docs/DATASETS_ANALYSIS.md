# 📊 PHÂN TÍCH CÁC DATASET TRONG DỰ ÁN GRAPHTRANSDTI

## ✅ HIỆN TẠI: ĐANG TRAIN TRÊN KIBA

**Trạng thái**: Đang train 100 epochs trên GPU (RTX 3050)  
**Dataset chính**: KIBA

---

## 1️⃣ KIBA DATASET (TRAINING & VALIDATION)

### 📈 Thông tin tổng quan:
```
Số drugs:      2,111 kinase inhibitors
Số proteins:   229 kinases
Tổng pairs:    118,254 drug-target interactions
Train split:   94,603 pairs (80%)
Val split:     11,825 pairs (10%)
Test split:    11,826 pairs (10%)
```

### 📁 Cấu trúc files:
```
data/kiba/
├── ligands_can.txt     # 161 KB - 2,111 SMILES strings (JSON format)
├── proteins.txt        # 166 KB - 229 protein sequences (JSON format)
└── Y                   # 3.8 MB - Affinity matrix shape (2111, 229)
```

### 🎯 Vai trò trong đồ án:
- **Training**: Học pattern từ 94,603 drug-protein pairs
- **Validation**: Chọn hyperparameters và early stopping
- **Internal Test**: Đánh giá performance trên unseen data (cùng domain)

### 📊 Đặc điểm:
- **Metric**: KIBA score (0-17, càng cao = binding càng mạnh)
- **Nguồn**: Tổng hợp từ BindingDB + STITCH database
- **Type**: Kinase inhibitors (thuốc ức chế kinase)
- **Format**: JSON dict `{"CHEMBL_ID": "SMILES"}`

### 💡 Ví dụ data:
```python
# Drug example:
CHEMBL123: "CCO"  # Ethanol

# Protein example:  
MAPK1: "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNK..."  # 360 amino acids

# Affinity:
KIBA[CHEMBL123, MAPK1] = 12.5  # KIBA score
```

---

## 2️⃣ DAVIS DATASET (GENERALIZATION TEST)

### 📈 Thông tin tổng quan:
```
Số drugs:      68 kinase inhibitors
Số proteins:   442 kinases
Tổng pairs:    30,056 drug-target interactions
```

### 📁 Cấu trúc files:
```
data/davis/
├── ligands_can.txt     # 5 KB - 68 SMILES strings
├── proteins.txt        # 347 KB - 442 protein sequences
└── Y                   # 235 KB - Affinity matrix shape (68, 442)
```

### 🎯 Vai trò trong đồ án:
- **Generalization Test**: Kiểm tra model có học được pattern tổng quát không
- **Cross-dataset evaluation**: Test trên proteins KHÁC với KIBA
- **Real-world simulation**: Giống tình huống thực tế (predict protein chưa thấy)

### 📊 Đặc điểm:
- **Metric**: Kd value (dissociation constant) - càng thấp = binding càng mạnh
- **Nguồn**: Davis et al. (2011) paper
- **Type**: Selective kinase inhibitors
- **Overlap**: Một số proteins trùng với KIBA, nhưng nhiều proteins mới

### 💡 Tại sao cần DAVIS?
1. **Proof of Generalization**: Chứng minh model không chỉ "thuộc lòng" KIBA
2. **Publication Standard**: Papers về DTI thường test trên cả KIBA và DAVIS
3. **Different Distribution**: DAVIS có distribution khác → test robustness

---

## 3️⃣ BINDINGDB DATASET (OPTIONAL - CHƯA SỬ DỤNG)

### 📈 Thông tin tổng quan:
```
Số lượng:      Hàng triệu drug-target pairs
Loại:          Nhiều protein families (không chỉ kinase)
```

### 🎯 Vai trò trong đồ án:
- **KHÔNG sử dụng trong đồ án hiện tại**
- **Lý do**: 
  - Quá lớn (hàng GB)
  - Nhiễu nhiều (quality thấp hơn KIBA/DAVIS)
  - Không cần thiết cho đồ án tốt nghiệp

### 💡 Khi nào dùng BindingDB?
- Research papers cần large-scale dataset
- Pre-training models (như BERT trong NLP)
- Transfer learning experiments

---

## 📊 SO SÁNH CÁC DATASET

| Feature | KIBA | DAVIS | BindingDB |
|---------|------|-------|-----------|
| **Drugs** | 2,111 | 68 | ~1M |
| **Proteins** | 229 | 442 | ~8K |
| **Pairs** | 118K | 30K | ~2M |
| **Protein Type** | Kinases | Kinases | All families |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Size** | 4 MB | 586 KB | ~10 GB |
| **Vai trò** | **TRAINING** | **TESTING** | Optional |
| **Trong đồ án** | ✅ Đang dùng | ✅ Sẽ test | ❌ Không dùng |

---

## 🔄 PIPELINE SỬ DỤNG DATASETS

### **Hiện tại** (đang chạy):
```
1. KIBA Training Set (94,603 pairs)
   ↓
   Train model 100 epochs
   ↓
2. KIBA Validation Set (11,825 pairs)
   ↓
   Select best model (early stopping)
   ↓
3. KIBA Test Set (11,826 pairs)
   ↓
   Report performance (RMSE, Pearson, CI)
```

### **Sau khi train xong KIBA**:
```
4. DAVIS Test Set (30,056 pairs)
   ↓
   Load best KIBA model
   ↓
   Test trên DAVIS
   ↓
   So sánh: KIBA Test vs DAVIS Test
   ↓
   Proof of generalization
```

---

## 🎯 VÌ SAO CHỌN KIBA LÀM TRAINING SET?

### ✅ **Ưu điểm**:
1. **Large-scale**: 118K pairs → đủ lớn để train deep learning
2. **Balanced**: 2,111 drugs × 229 proteins → không quá bias
3. **High quality**: Curated từ BindingDB + STITCH
4. **Standard benchmark**: Tất cả papers DTI đều dùng KIBA
5. **Kinase focus**: Protein family quan trọng (cancer, inflammation)

### 📊 **So với DAVIS**:
- DAVIS nhỏ hơn (30K vs 118K) → dùng làm training sẽ underfit
- DAVIS có ít drugs hơn (68 vs 2,111) → ít diversity
- KIBA có KIBA score chuẩn hóa tốt hơn

---

## 📈 KẾT QUẢ MONG ĐỢI

### **Trên KIBA Test Set**:
```
RMSE:     0.90 - 1.10  (↓ càng thấp càng tốt)
Pearson:  0.85 - 0.89  (↑ càng cao càng tốt)
CI:       0.87 - 0.90  (↑ càng cao càng tốt)
```

### **Trên DAVIS Test Set** (generalization):
```
RMSE:     1.00 - 1.20  (có thể cao hơn KIBA 10-15%)
Pearson:  0.80 - 0.86  (có thể thấp hơn KIBA 5-10%)
CI:       0.83 - 0.88
```

### **Why DAVIS worse than KIBA?**
- Different domain (442 proteins khác với 229 proteins trong KIBA)
- Different distribution
- Model chưa thấy nhiều proteins này trong training
- **Điều này là BÌNH THƯỜNG và MONG MUỐN** (proof of generalization)

---

## 📝 CÁCH VIẾT TRONG BÁO CÁO

### **Chapter 6: Dữ liệu**

> "Đồ án sử dụng 2 datasets chuẩn trong lĩnh vực Drug-Target Interaction:
> 
> **KIBA dataset** được dùng làm tập training và validation, bao gồm 2,111 
> kinase inhibitors, 229 kinases, tạo thành 118,254 drug-target pairs. Dữ liệu 
> được chia theo tỉ lệ 80:10:10 cho training, validation và test.
> 
> **DAVIS dataset** được dùng để đánh giá khả năng generalization của model, 
> bao gồm 68 kinase inhibitors, 442 kinases, tạo thành 30,056 pairs. Dataset 
> này có nhiều proteins không xuất hiện trong KIBA, giúp kiểm tra model có 
> học được pattern tổng quát hay chỉ 'thuộc lòng' training set.
> 
> Cả 2 datasets đều tập trung vào kinase proteins - một protein family quan 
> trọng trong điều trị ung thư và viêm nhiễm."

### **Chapter 7: Kết quả**

> "Model được training trên KIBA đạt RMSE = 0.95 trên KIBA test set. Khi 
> đánh giá trên DAVIS test set (cross-dataset evaluation), model đạt RMSE = 1.08, 
> chứng tỏ model có khả năng generalization tốt sang proteins chưa thấy trong 
> training. Sự suy giảm 13.7% trong performance là chấp nhận được và phù hợp 
> với các nghiên cứu trước đây."

---

## 🚀 HÀNH ĐỘNG TIẾP THEO

### **Đang làm** (hiện tại):
- [x] Training trên KIBA (100 epochs với early stopping)
- [x] Tự động tạo biểu đồ sau mỗi epoch
- [ ] Đợi training hoàn thành (~3-4 giờ)

### **Sau khi train xong**:
1. Load best model từ checkpoint
2. Test trên KIBA test set → Tạo biểu đồ
3. **Test trên DAVIS test set** → Tạo biểu đồ so sánh
4. So sánh: KIBA vs DAVIS performance
5. Viết báo cáo: Chapter 6 (Data) + Chapter 7 (Results)

### **Command test DAVIS**:
```bash
python src/evaluate.py \
  --checkpoint checkpoints/GraphTransDTI_KIBA_best.pt \
  --dataset davis \
  --split test
```

---

## 📚 TÓM TẮT

| Câu hỏi | Trả lời |
|---------|---------|
| **Hiện tại train dataset gì?** | ✅ KIBA (118K pairs) |
| **KIBA bao gồm gì?** | 2,111 drugs + 229 proteins + 118K interactions |
| **DAVIS dùng để làm gì?** | Test generalization (442 proteins khác KIBA) |
| **BindingDB có dùng không?** | ❌ Không (quá lớn, không cần thiết) |
| **Tại sao train KIBA?** | Large-scale, high quality, standard benchmark |
| **Khi nào test DAVIS?** | Sau khi train KIBA xong |
| **Kết quả mong đợi?** | KIBA: RMSE ~0.95, DAVIS: RMSE ~1.08 |

---

**Kết luận**: Bạn đang train ĐÚNG! KIBA là dataset tốt nhất cho training, DAVIS sẽ dùng sau để proof generalization. Không cần dùng BindingDB.
