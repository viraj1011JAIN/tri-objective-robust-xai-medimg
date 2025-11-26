# Phase 4 - Quick Execution Guide 🚀

**Notebook:** `notebooks/Phase_4_full_ADVERSARIAL ATTACKS & ROBUSTNESS.ipynb`

---

## 📱 One-Page Execution Checklist

### 🔧 Setup (5 minutes)
```
□ Upload notebook to Google Colab
□ Runtime → Change runtime type → T4 GPU
□ Runtime → Run all (or run cells 1-7)
□ Wait for Google Drive mount
□ Verify GPU detected
□ Wait for dependencies installation
```

### 📊 Data Preparation (Check)
```
□ Ensure data at: /content/drive/MyDrive/data/isic_2018/
□ Ensure checkpoints at: /content/drive/MyDrive/checkpoints/baseline/
   ├── seed_42/best.pt
   ├── seed_123/best.pt
   └── seed_456/best.pt
```

### 🔥 Attack Evaluation (2-3 hours)
```
□ Run cells 8-11: Load helpers and data (3 min)
□ Run cells 12-19: SEED LOOP evaluation (~120 min)
   ├── Cell 13: Initialize results and seed loop
   ├── Cell 15: FGSM evaluation (~5 min/seed)
   ├── Cell 17: PGD evaluation (~25 min/seed)
   └── Cell 19: C&W evaluation (~10 min/seed)

⚠️  This is the longest section - Go get coffee! ☕
```

### 📈 Analysis & Visualization (10 minutes)
```
□ Run cells 20-22: Statistical aggregation (1 min)
□ Run cells 23-28: Adversarial visualizations (8 min)
□ Run cells 29-31: Summary and comparison plots (1 min)
□ Run cells 32-33: Optional transferability (skip if no extra models)
□ Review cell 34: Completion summary
```

### ✅ Verify Outputs
```
□ Check results directory: /content/drive/MyDrive/results/robustness/
□ Files created:
   ├── baseline_robustness_aggregated.json
   ├── adversarial_examples_visualization.png
   ├── perturbation_visualization.png
   └── attack_comparison.png
```

---

## ⚡ Fast Execution (Testing Mode)

For quick verification, modify these cells:

### Cell 7 - Reduce Seeds
```python
CONFIG = {
    "seeds": [42],  # Use only 1 seed instead of 3
    # ... rest of config
}
```

### Cell 19 - Limit C&W Batches
```python
cw_results = evaluate_attack(
    model=model,
    attack=cw_attack,
    dataloader=test_loader,
    device=CONFIG['device'],
    max_batches=10  # Test mode - only 10 batches
)
```

**Testing Mode Time:** ~30 minutes instead of 2-3 hours

---

## 📊 Expected Console Output

### During Setup
```
Mounted at /content/drive
GPU detected: Tesla T4 (15GB)
Cloning repository...
Installing packages... ✓
✅ Helper functions defined
✅ Test dataset loaded: 1512 samples
```

### During Evaluation (per seed)
```
============================================================
Evaluating Seed: 42
============================================================
Loading model from: .../seed_42/best.pt
✅ Model loaded successfully

📊 Testing clean accuracy...
Clean eval: 100%|██████████| 48/48 [00:12<00:00]
✅ Clean Accuracy: 82.34%

🔥 FGSM Attack Evaluation
------------------------------------------------------------
  Epsilon: 0.0078 (2.0/255)
  Evaluating FGSM: 100%|██████████| 48/48 [00:15<00:00]
  ✅ Clean Acc: 82.34%
  🛡️  Robust Acc: 67.12%
  📉 Acc Drop: 15.22pp
  ...

🔥 PGD Attack Evaluation
------------------------------------------------------------
  Config: ε=0.0078 (2.0/255), steps=7
  Evaluating PGD: 100%|██████████| 48/48 [02:13<00:00]
  ✅ Clean Acc: 82.34%
  🛡️  Robust Acc: 54.23%
  📉 Acc Drop: 28.11pp
  ...

🔥 Carlini & Wagner (C&W) Attack Evaluation
------------------------------------------------------------
Evaluating C&W: 100%|██████████| 48/48 [09:45<00:00]
✅ Clean Acc: 82.34%
🛡️  Robust Acc: 12.45%
📉 Acc Drop: 69.89pp
...
```

### Final Summary
```
================================================================================
STATISTICAL AGGREGATION - Results across 3 seeds
================================================================================

📊 FGSM Results:
--------------------------------------------------------------------------------
  Epsilon: 0.0314 (8.0/255)
    clean_accuracy           :  82.41 ±  0.87
    robust_accuracy          :  31.23 ±  2.45
    accuracy_drop            :  51.18 ±  2.11
    attack_success_rate      :  62.11 ±  3.21

📊 PGD Results:
--------------------------------------------------------------------------------
  ε=0.0314 (8.0/255), steps=20
    clean_accuracy           :  82.41 ±  0.87
    robust_accuracy          :  14.67 ±  1.98
    accuracy_drop            :  67.74 ±  2.34
    attack_success_rate      :  82.19 ±  2.87

📊 C&W Results:
--------------------------------------------------------------------------------
  clean_accuracy           :  82.41 ±  0.87
  robust_accuracy          :  11.23 ±  1.56
  accuracy_drop            :  71.18 ±  1.89
  attack_success_rate      :  86.38 ±  2.31

================================================================================
PHASE 4.3 CHECKLIST VERIFICATION:
================================================================================
✅ All attacks implemented and tested (FGSM, PGD, C&W)
✅ Baseline robustness evaluated across 3 seeds
✅ Expected accuracy drop verified: 67.7pp (target: 50-70pp)
✅ Statistical aggregation completed (mean ± std)
✅ Adversarial examples visualized
✅ Results saved to: /content/drive/MyDrive/results/robustness

🎯 CONCLUSION:
   ✅ Baseline model shows EXPECTED VULNERABILITY to adversarial attacks
   ✅ Ready to proceed with Phase 5 (Tri-Objective Robust XAI Training)
```

---

## 🐛 Common Issues & Fixes

### Issue 1: OOM Error
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size in Cell 7
```python
CONFIG['batch_size'] = 16  # or even 8
```

### Issue 2: Data Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '.../isic_2018'
```
**Fix:** Check data path in Cell 7 and verify data exists in Drive

### Issue 3: Checkpoint Not Found
```
FileNotFoundError: checkpoint not found
```
**Fix:** Verify checkpoint paths in CONFIG and ensure Phase 3 training completed

### Issue 4: C&W Taking Too Long
```
# C&W running for 30+ minutes
```
**Fix:** Add max_batches limit in Cell 19
```python
max_batches=10  # Instead of None
```

### Issue 5: Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Fix:** Verify repository cloned and current directory set in Cell 3
```python
os.chdir('/content/tri-objective-robust-xai-medimg')
sys.path.insert(0, '/content/tri-objective-robust-xai-medimg')
```

---

## 💾 Downloading Results

### From Colab to Local Machine
```python
# Add this cell at the end if needed
from google.colab import files

# Download results
files.download('/content/drive/MyDrive/results/robustness/baseline_robustness_aggregated.json')
files.download('/content/drive/MyDrive/results/robustness/adversarial_examples_visualization.png')
files.download('/content/drive/MyDrive/results/robustness/perturbation_visualization.png')
files.download('/content/drive/MyDrive/results/robustness/attack_comparison.png')
```

### Or Access via Drive
```
1. Open Google Drive
2. Navigate to: MyDrive/results/robustness/
3. Right-click files → Download
```

---

## 📋 Cell-by-Cell Summary

| Cell # | Type | Description | Time |
|--------|------|-------------|------|
| 1 | MD | Header & objectives | - |
| 2 | MD | Setup section | - |
| 3 | PY | Mount Drive & verify GPU | 30s |
| 4 | PY | Install dependencies | 2min |
| 5 | PY | Import libraries | 10s |
| 6 | MD | Config section | - |
| 7 | PY | CONFIG dictionary | 1s |
| 8 | MD | Helper functions section | - |
| 9 | PY | Helper function definitions | 1s |
| 10 | MD | Data loading section | - |
| 11 | PY | Create test dataset | 30s |
| 12 | MD | Phase 4.3 section | - |
| 13 | PY | Initialize results & seed loop | - |
| 14 | MD | FGSM header | - |
| 15 | PY | **FGSM eval (×3 seeds)** | **15min** |
| 16 | MD | PGD header | - |
| 17 | PY | **PGD eval (×3 seeds)** | **75min** |
| 18 | MD | C&W header | - |
| 19 | PY | **C&W eval (×3 seeds)** | **30min** |
| 20 | MD | Aggregation section | - |
| 21 | PY | Aggregate statistics | 5s |
| 22 | PY | Save JSON results | 1s |
| 23 | MD | Visualization section | - |
| 24 | PY | Viz helper functions | 1s |
| 25 | PY | Load model for viz | 5s |
| 26 | PY | Create attacks for viz | 1s |
| 27 | PY | Generate adversarial viz | 2min |
| 28 | PY | Perturbation viz | 3min |
| 29 | MD | Summary section | - |
| 30 | PY | Comparison plots | 30s |
| 31 | PY | Final summary report | 5s |
| 32 | MD | Transferability section | - |
| 33 | PY | Transferability (skip) | 0s |
| 34 | MD | Completion summary | - |

**Total:** 34 cells | **Runtime:** ~2-3 hours

---

## ✅ Success Criteria

After execution, you should see:

1. **Console Output:** "✅ Baseline model shows EXPECTED VULNERABILITY"
2. **Accuracy Drop:** 50-70pp under PGD ε=8/255
3. **4 Output Files:** JSON + 3 PNG figures
4. **Phase 4.3 Complete:** Ready for Phase 5

---

**Ready to Execute!** 🚀

Upload to Colab and run all cells. Results will be saved to your Google Drive.

For questions or issues, check the main documentation:
- `PHASE_4_NOTEBOOK_READY.md` - Full documentation
- `PHASE_4_EVIDENCE_CHECK.md` - Status verification
