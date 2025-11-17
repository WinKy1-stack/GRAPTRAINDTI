# Hướng dẫn tải dữ liệu cho GraphTransDTI

## Tổng quan

GraphTransDTI sử dụng 2 datasets chính:
- **KIBA**: Training và validation
- **DAVIS**: Testing (đánh giá khả năng generalization)

Cả 2 datasets đều sử dụng **format pickle** từ DeepDTA repository.

---

## 1. KIBA Dataset

### Thông tin
- **Số lượng drugs**: 2,111 kinase inhibitors
- **Số lượng proteins**: 229 kinases  
- **Số lượng pairs**: 118,254 drug-target interactions
- **Metric**: KIBA score (0-17, càng cao càng mạnh)
- **Source**: BindingDB + STITCH

### Cấu trúc files

```
data/kiba/
├── ligands_can.txt     # 2,111 SMILES (1 dòng = 1 drug)
├── proteins.txt        # 229 protein sequences (1 dòng = 1 protein)
└── Y                   # Affinity matrix (2111 x 229) - pickle format
```

### Tải dữ liệu

**PowerShell**:
```powershell
# Tạo thư mục
New-Item -ItemType Directory -Path "data/kiba" -Force
cd data/kiba

# Tải 3 files
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/ligands_can.txt" -OutFile "ligands_can.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/proteins.txt" -OutFile "proteins.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/Y" -OutFile "Y"

cd ../..
```

**Bash** (Linux/Mac):
```bash
mkdir -p data/kiba
cd data/kiba

wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/ligands_can.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/proteins.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba/Y

cd ../..
```

---

## 2. DAVIS Dataset

### Thông tin
- **Số lượng drugs**: 68 kinase inhibitors
- **Số lượng proteins**: 442 kinases
- **Số lượng pairs**: 30,056 drug-target interactions  
- **Metric**: Kd (dissociation constant in nM)
- **Source**: Davis et al. 2011

### Cấu trúc files

```
data/davis/
├── ligands_can.txt     # 68 SMILES
├── proteins.txt        # 442 protein sequences
└── Y                   # Affinity matrix (68 x 442) - pickle format
```

### Tải dữ liệu

**PowerShell**:
```powershell
# Tạo thư mục
New-Item -ItemType Directory -Path "data/davis" -Force
cd data/davis

# Tải 3 files
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/ligands_can.txt" -OutFile "ligands_can.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/proteins.txt" -OutFile "proteins.txt"
Invoke-WebRequest -Uri "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/Y" -OutFile "Y"

cd ../..
```

**Bash**:
```bash
mkdir -p data/davis
cd data/davis

wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/ligands_can.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/proteins.txt
wget https://github.com/hkmztrk/DeepDTA/raw/master/data/davis/Y

cd ../..
```

---

## 3. Kiểm tra dữ liệu

Sau khi download, verify rằng data đã load thành công:

```powershell
cd src

# Test KIBA loader
python dataloader/kiba_loader.py

# Test DAVIS loader  
python dataloader/davis_loader.py

cd ..
```

**Expected output**:
```
[INFO] Loaded 2111 drugs, 229 proteins
[INFO] Affinity matrix shape: (2111, 229)
[INFO] Created 118254 valid drug-protein pairs
[INFO] KIBA TRAIN dataset loaded: 94603 pairs
```

---

## 4. Cấu trúc thư mục hoàn chỉnh

Sau khi tải xong, cấu trúc dự kiến:

```
GraphTransDTI/
├── data/
│   ├── kiba/
│   │   ├── ligands_can.txt
│   │   ├── proteins.txt
│   │   └── Y
│   ├── davis/
│   │   ├── ligands_can.txt
│   │   ├── proteins.txt
│   │   └── Y
│   └── DATA_DOWNLOAD_GUIDE.md (file này)
├── src/
│   ├── dataloader/
│   │   ├── featurizer.py
│   │   ├── kiba_loader.py
│   │   └── davis_loader.py
│   ├── train.py
│   └── ...
└── config.yaml
```

---

## 5. Chi tiết format

### ligands_can.txt
- Mỗi dòng là 1 SMILES string
- Canonical SMILES (chuẩn hóa)
- Ví dụ:
```
CCO
CC(=O)O
c1ccccc1
...
```

### proteins.txt
- Mỗi dòng là 1 protein sequence
- Amino acid one-letter code
- Ví dụ:
```
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKK
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSYQGFSLFPEDFFGVNL
...
```

### Y (pickle file)
- Python pickle format (numpy array)
- Shape: (num_drugs, num_proteins)
- Giá trị:
  - **KIBA**: KIBA score (0-17), continuous
  - **DAVIS**: Kd value (nM), continuous
  - **NaN**: No interaction data
- Load bằng:
```python
import pickle
with open('Y', 'rb') as f:
    affinity = pickle.load(f, encoding='latin1')
```

---

## 6. Troubleshooting

### Lỗi: FileNotFoundError
**Nguyên nhân**: Files chưa download hoặc sai path

**Giải pháp**:
```powershell
# Kiểm tra files có tồn tại không
Test-Path "data/kiba/ligands_can.txt"
Test-Path "data/kiba/proteins.txt"
Test-Path "data/kiba/Y"
```

### Lỗi: Download failed
**Nguyên nhân**: GitHub rate limit hoặc network issue

**Giải pháp**:
```powershell
# Enable TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Thử lại
Invoke-WebRequest -Uri "..." -OutFile "..."
```

### Lỗi: Cannot unpickle
**Nguyên nhân**: File Y là Python 2 pickle

**Giải pháp**: Code đã tự động xử lý bằng `encoding='latin1'`

### Lỗi: Wrong matrix shape
**Nguyên nhân**: Download chưa hoàn chỉnh

**Giải pháp**:
```powershell
# Xóa và download lại
Remove-Item "data/kiba/Y" -Force
Invoke-WebRequest -Uri "..." -OutFile "Y"
```

---

## 7. Dataset Statistics

### KIBA
| Metric | Value |
|--------|-------|
| Drugs | 2,111 |
| Proteins | 229 |
| Valid pairs | 118,254 |
| Matrix sparsity | ~75% |
| Affinity range | 0 - 17 (KIBA score) |
| File size | ~2 MB (pickle) |

### DAVIS
| Metric | Value |
|--------|-------|
| Drugs | 68 |
| Proteins | 442 |
| Valid pairs | 30,056 |
| Matrix sparsity | ~99% |
| Affinity range | 5 - 10,000 nM (Kd) |
| File size | ~500 KB (pickle) |

---

## 8. References

### Papers
- **KIBA**: Tang et al. (2014). "Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis." *Journal of Chemical Information and Modeling*.
- **DAVIS**: Davis et al. (2011). "Comprehensive analysis of kinase inhibitor selectivity." *Nature Biotechnology*.

### Code
- **DeepDTA**: Öztürk et al. (2018) - https://github.com/hkmztrk/DeepDTA

### Data sources
- **BindingDB**: https://www.bindingdb.org/
- **STITCH**: http://stitch.embl.de/

---

## 9. Sử dụng

Sau khi download xong:

```powershell
# 1. Activate environment
.\venv\Scripts\Activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Test dataloader
cd src
python dataloader/kiba_loader.py
python dataloader/davis_loader.py

# 4. Train model
python train.py

cd ..
```

---

**Lưu ý**: Dữ liệu này chỉ dùng cho mục đích nghiên cứu và giáo dục theo license của các datasets gốc.
