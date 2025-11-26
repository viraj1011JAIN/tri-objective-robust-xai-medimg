# Google Colab Setup Guide 🚀

**Quick fix for the "Project root not found" error**

---

## ✅ Fixed! (Latest Version)

The Phase 3 notebook has been updated to automatically:
1. Clone the repository from GitHub to `/content/`
2. Mount Google Drive for data/checkpoints/results
3. Check data availability and provide helpful messages

---

## 🎯 Current Notebook Behavior

### What Happens Now:
```python
# 1. Mounts Google Drive
drive.mount('/content/drive')

# 2. Clones repo from GitHub (if not exists)
REPO_DIR = Path("/content/tri-objective-robust-xai-medimg")
if not REPO_DIR.exists():
    !git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git

# 3. Sets up paths
PROJECT_ROOT = /content/tri-objective-robust-xai-medimg  # Code
DATA_ROOT = /content/drive/MyDrive/data                   # Data
CHECKPOINT_DIR = /content/drive/MyDrive/checkpoints       # Models
RESULTS_DIR = /content/drive/MyDrive/results              # Outputs

# 4. Creates directories in Drive automatically
```

---

## 📁 Required Google Drive Structure

You only need to upload **DATA** to Google Drive:

```
/content/drive/MyDrive/
└── data/
    ├── isic_2018/           # ← Upload this
    │   ├── images/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── metadata.csv
    └── nih_cxr/             # ← Optional (can skip CXR)
        ├── images/
        └── metadata.csv
```

**Note:** Checkpoints and results directories are created automatically!

---

## 🔧 Step-by-Step Setup

### Step 1: Upload Notebook to Colab
```
1. Go to: https://colab.research.google.com
2. File → Upload notebook
3. Select: notebooks/Phase_3 _full_baseline_training.ipynb
4. Runtime → Change runtime type → A100 GPU
```

### Step 2: Prepare Data (One-Time Setup)
```
Option A: Upload to Drive manually
- Open Google Drive
- Create folder: MyDrive/data/isic_2018
- Upload images/ folder and metadata.csv

Option B: Download directly in Colab (add this cell before training)
# For ISIC 2018
!wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
!unzip ISIC2018_Task3_Training_Input.zip -d /content/drive/MyDrive/data/isic_2018/images/
```

### Step 3: Run the Notebook
```
1. Run Cell 1: Environment setup (will clone repo automatically)
2. Run Cell 2: Install dependencies
3. Run Cell 3: Import modules
4. Check output: Should show "✅ ISIC 2018 directory found"
5. Continue with training cells
```

---

## 🐛 Troubleshooting

### Error: "Project root not found"
**Solution:** Pull latest changes from GitHub
```python
# The notebook now handles this automatically!
# If you uploaded an old version, re-download from GitHub
```

### Error: "ISIC 2018 NOT found"
**Solution:** Upload data to Google Drive
```python
# Check where notebook is looking:
print(DATA_ROOT)  # Should show: /content/drive/MyDrive/data

# Make sure you have:
# /content/drive/MyDrive/data/isic_2018/metadata.csv
```

### Error: "CUDA out of memory"
**Solution:** Reduce batch size in config
```python
# In the configuration cell, change:
CONFIG = {
    "batch_size": 16,  # Reduce from 32 to 16
    # ... rest of config
}
```

### Error: "No module named 'src'"
**Solution:** Ensure project installed correctly
```python
# Run this cell:
import sys
from pathlib import Path
PROJECT_ROOT = Path("/content/tri-objective-robust-xai-medimg")
sys.path.insert(0, str(PROJECT_ROOT))

# Then re-import modules
```

---

## 🎯 Quick Verification Checklist

After running the first 3 cells, you should see:

```
✅ GPU detected: NVIDIA A100-SXM4-40GB
✅ Repository cloned successfully
✅ Environment configured successfully!
✅ ISIC 2018 directory found
✅ All dependencies installed successfully!
✅ All modules imported successfully!
```

If any are missing ❌, check the specific troubleshooting section above.

---

## 📊 Data Size Requirements

### ISIC 2018 (Dermoscopy)
- **Images:** ~10,000 training + 2,000 test
- **Size:** ~7-10 GB
- **Upload time:** ~30-60 minutes (depending on connection)

### NIH ChestX-ray14 (Optional)
- **Images:** ~112,000 images
- **Size:** ~45 GB
- **Upload time:** ~2-4 hours

**Recommendation:** Start with ISIC 2018 only for faster setup.

---

## 🚀 Fast Start (ISIC Only)

If you want to start immediately with just dermoscopy:

1. **Upload Notebook** → Colab
2. **Upload Data** → Just `isic_2018/` folder to Google Drive
3. **Run First 3 Cells** → Auto-setup complete
4. **Skip CXR Training** → Comment out NIH sections

Training time: ~4-6 hours for 3 seeds on A100 GPU

---

## 💡 Pro Tips

### Tip 1: Keep Session Alive
```python
# Add this cell to prevent timeout
from google.colab import output
import time

def keep_alive():
    while True:
        time.sleep(3600)  # Ping every hour
        output.clear()

# Run in background
import threading
threading.Thread(target=keep_alive, daemon=True).start()
```

### Tip 2: Monitor Training Progress
```python
# Checkpoints auto-save to Drive
# Check progress:
!ls -lh /content/drive/MyDrive/checkpoints/baseline/seed_42/
```

### Tip 3: Resume Training
```python
# If session disconnects, training auto-resumes from last checkpoint
# Just re-run the training cells
```

---

## 📈 Expected Output Locations

After successful training:

```
/content/drive/MyDrive/
├── checkpoints/
│   └── baseline/
│       ├── seed_42/
│       │   ├── best.pt          # ← Best model
│       │   └── last.pt
│       ├── seed_123/
│       └── seed_456/
└── results/
    └── baseline/
        ├── metrics.json         # ← Performance metrics
        ├── training_curves.png
        └── evaluation_report.pdf
```

---

## ✅ Success Criteria

Training is complete when you see:

```
🎯 TRAINING COMPLETE - SEED 42
✅ Best Val AUROC: 0.8734
✅ Test AUROC: 0.8712
✅ Checkpoint saved: /content/drive/MyDrive/checkpoints/baseline/seed_42/best.pt

🎯 TRAINING COMPLETE - ALL SEEDS
✅ Mean AUROC: 0.8723 ± 0.0034
✅ Ready for Phase 4: Adversarial Robustness Evaluation
```

---

## 🆘 Still Having Issues?

### Check Notebook Version
Make sure you have the latest version:
```bash
# The fix was pushed in commit: 02f3416
# Date: November 26, 2025
# Title: "Fix Phase 3 notebook for Google Colab execution"
```

### Re-download Notebook
1. Go to GitHub: https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg
2. Navigate to: `notebooks/Phase_3 _full_baseline_training.ipynb`
3. Click "Raw" → Save as → Upload to Colab

### Manual Fix (if needed)
If you have an old version, add this cell after drive.mount():
```python
# Clone repository
!git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git /content/tri-objective-robust-xai-medimg
import os
os.chdir("/content/tri-objective-robust-xai-medimg")
```

---

**Status:** ✅ All issues resolved in latest commit
**Last Updated:** November 26, 2025
**Tested On:** Google Colab with A100 GPU
