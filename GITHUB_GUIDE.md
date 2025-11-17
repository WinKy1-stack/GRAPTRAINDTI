# Hướng dẫn đưa dự án lên GitHub

## ✅ Đã hoàn thành

1. ✅ Đã khởi tạo git repository
2. ✅ Đã tạo `.gitignore` (loại trừ data, checkpoints lớn)
3. ✅ Đã tạo `README.md` (mô tả dự án đầy đủ)
4. ✅ Đã tạo `LICENSE` (MIT License)
5. ✅ Đã commit lần đầu (42 files)

## 🚀 Bước tiếp theo

### 1. Tạo repository trên GitHub

Truy cập: https://github.com/new

**Cài đặt:**
- Repository name: `GraphTransDTI` (hoặc tên khác)
- Description: `Graph Transformer for Drug-Target Interaction Prediction - Graduation Thesis`
- Visibility: **Public** (để chia sẻ) hoặc **Private** (nếu muốn giữ kín)
- ⚠️ **KHÔNG** chọn "Add README" (đã có rồi)
- ⚠️ **KHÔNG** chọn "Add .gitignore" (đã có rồi)
- ⚠️ **KHÔNG** chọn "Choose a license" (đã có rồi)

Nhấn **"Create repository"**

### 2. Liên kết với GitHub và push

Sau khi tạo repository, GitHub sẽ hiện hướng dẫn. Chạy lệnh sau:

```powershell
# Thêm remote origin (thay YOUR_USERNAME bằng username GitHub của bạn)
git remote add origin https://github.com/YOUR_USERNAME/GraphTransDTI.git

# Xác nhận branch là main
git branch -M main

# Push code lên GitHub
git push -u origin main
```

### 3. Xác thực GitHub

Khi push lần đầu, Windows sẽ hiện cửa sổ xác thực:
- Chọn **"Sign in with your browser"**
- Đăng nhập GitHub
- Cho phép truy cập

Hoặc dùng Personal Access Token:
```powershell
# Tạo token tại: https://github.com/settings/tokens
# Chọn: repo (full control of private repositories)
# Copy token và dùng thay password khi push
```

### 4. Xác nhận đã push thành công

```powershell
git remote -v
# Kết quả:
# origin  https://github.com/YOUR_USERNAME/GraphTransDTI.git (fetch)
# origin  https://github.com/YOUR_USERNAME/GraphTransDTI.git (push)

git log --oneline
# Hiện commit history
```

## 📦 Những gì được đưa lên GitHub

### ✅ Code và Documentation (42 files, ~11K lines)
- ✅ Model implementation (`src/models/`)
- ✅ Data loaders (`src/dataloader/`)
- ✅ Training scripts (`src/train.py`)
- ✅ Evaluation tools (`test_davis_normalized.py`)
- ✅ Visualization utilities (`src/visualize_results.py`)
- ✅ Documentation (`docs/*.md`)
- ✅ README, LICENSE, .gitignore

### ❌ KHÔNG đưa lên (theo .gitignore)
- ❌ Virtual environment (`venv/`)
- ❌ Data files (`data/kiba/`, `data/davis/`)
- ❌ Model checkpoints (`checkpoints/*.pt`) - quá lớn (>500MB)
- ❌ Results (`results/*.png`)
- ❌ Cache files (`__pycache__/`)

## 🎯 Sau khi push

### 1. Thêm Releases (optional)

Để chia sẻ trained model:

1. Truy cập: `https://github.com/YOUR_USERNAME/GraphTransDTI/releases/new`
2. Tag version: `v1.0.0`
3. Release title: `GraphTransDTI v1.0.0 - Initial Release`
4. Upload files:
   - `GraphTransDTI_KIBA_best.pt` (model checkpoint)
   - `results_summary.json` (metrics)
   - Sample plots

### 2. Cập nhật README

Sửa thông tin cá nhân trong `README.md`:

```markdown
## 👤 Author

**[Tên của bạn]**
- University: [Trường của bạn]
- Email: [email@example.com]
- Advisor: [Tên giáo viên hướng dẫn]
```

Sau đó commit và push:
```powershell
git add README.md
git commit -m "Update author information"
git push
```

### 3. Thêm Topics (tags)

Trên trang GitHub repository → Settings → About → Topics:
- `drug-discovery`
- `deep-learning`
- `pytorch`
- `graph-neural-networks`
- `bioinformatics`
- `drug-target-interaction`
- `graduation-thesis`

### 4. GitHub Pages (optional)

Để host documentation:
1. Settings → Pages
2. Source: Deploy from branch → `main` → `/docs`
3. Save

Documentation sẽ có tại: `https://YOUR_USERNAME.github.io/GraphTransDTI/`

## 🔧 Lệnh Git hữu ích

```powershell
# Xem trạng thái
git status

# Thêm file mới/sửa đổi
git add .
git commit -m "Your message"
git push

# Xem lịch sử
git log --oneline --graph

# Tạo branch mới (để thử nghiệm)
git checkout -b experiment
git push -u origin experiment

# Quay về main
git checkout main

# Pull changes từ GitHub
git pull origin main

# Xóa file khỏi git nhưng giữ local
git rm --cached <file>
git commit -m "Remove file from git"
git push
```

## 📋 Checklist cuối cùng

Trước khi public repository:

- [ ] Đã thay đổi thông tin tác giả trong README.md
- [ ] Đã kiểm tra không có thông tin nhạy cảm (API keys, passwords)
- [ ] Đã test clone repository và chạy được
- [ ] Đã viết rõ hướng dẫn cài đặt
- [ ] Đã document các kết quả chính
- [ ] Đã thêm LICENSE (MIT hoặc tương tự)
- [ ] Đã thêm badges trong README (Python version, PyTorch, License)

## 🎓 Cho luận văn

### Clone instructions cho giáo viên/bạn bè:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/GraphTransDTI.git
cd GraphTransDTI

# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r src/requirements.txt

# Download data (xem DATA_DOWNLOAD_GUIDE.md)

# Download trained model từ Releases tab

# Test inference
python test_model.py
```

### Thêm vào slide thuyết trình:

```
GitHub Repository:
https://github.com/YOUR_USERNAME/GraphTransDTI

⭐ 42 files | 10,688+ lines of code
📊 8% RMSE improvement over baseline
🔬 Reproducible experiments
📖 Complete documentation
```

## 💡 Tips

1. **Commit thường xuyên**: Mỗi feature mới → 1 commit
2. **Message rõ ràng**: "Add cross-attention visualization" thay vì "update"
3. **Branch cho experiments**: Main branch giữ code stable
4. **README is your CV**: README tốt = dự án professional

## 🆘 Troubleshooting

**Lỗi: Permission denied**
```powershell
# Kiểm tra SSH key hoặc dùng HTTPS với token
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/GraphTransDTI.git
```

**Lỗi: Large files rejected**
```powershell
# Xóa file khỏi git history
git rm --cached checkpoints/*.pt
git commit --amend
git push -f
```

**Lỗi: Merge conflict**
```powershell
# Pull trước khi push
git pull origin main --rebase
git push
```

---

**Ready to push?** 🚀

Chạy lệnh:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/GraphTransDTI.git
git push -u origin main
```
